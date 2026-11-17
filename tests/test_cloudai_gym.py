# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from cloudai.configurator import Trajectory
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import (
    METRIC_ERROR,
    CloudAIGymEnv,
    CmdArgs,
    MetricValue,
    ObsLeafDescriptor,
    RewardOverrides,
    System,
    TestDefinition,
    TestRun,
    TestScenario,
)
from cloudai.systems.slurm import SlurmRunner, SlurmSystem


@pytest.mark.parametrize(
    "randomized, weights",
    [(False, None), (True, None), (True, [0.0, 1.0])],
    ids=["static", "uniform", "weighted"],
)
@pytest.mark.parametrize("node_action", [False, True], ids=["fixed-nodes", "tuned-nodes"])
def test_step_records_resolved_static_context_without_expanding_policy_observations(
    slurm_system: SlurmSystem,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    randomized: bool,
    weights: list[float] | None,
    node_action: bool,
) -> None:
    fixed_values = {
        "label": "001",
        "enabled": True,
        "optional": None,
        "empty_mapping": {},
        "empty_list": [],
        "mixed": ["001", 1, True, None, {}],
    }
    tdef = TestDefinition(
        name="context",
        description="context",
        test_template_name="context",
        cmd_args=CmdArgs.model_validate(
            {
                "paddle_width": [4, 8],
                "ball_speed": [1, 3] if randomized else 3,
                "nested": {"tuned": [10, 20], "fixed": fixed_values},
                "literal_list": ["keep", "both"],
            }
        ),
        env_params={"ball_speed": EnvParamSpec(weights=weights)} if randomized else {},
        dse_excluded_args=["cmd_args.nested.fixed", "cmd_args.literal_list"],
        extra_env_vars={"FIXED": "001", "TUNED": ["left", "right"]},
        extra_cmd_args={"--precision": "bf16"},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )
    tr = TestRun(name="context", test=tdef, num_nodes=[1, 2] if node_action else 2, nodes=[])
    scenario = TestScenario(name="context", test_runs=[tr])
    runner = SlurmRunner(mode="dry-run", system=slurm_system, test_scenario=scenario, output_path=tmp_path / "run")
    env = CloudAIGymEnv(tr, runner, RewardOverrides())
    monkeypatch.setattr(runner, "run", lambda: None)
    monkeypatch.setattr(env, "get_observation", lambda *, failed_metrics=None: {"default": 100.0})
    action = {"paddle_width": 4, "nested.tuned": 20, "extra_env_vars.TUNED": "right"}
    if node_action:
        action["NUM_NODES"] = 2

    initial_observation, initial_info = env.reset()
    observation, reward, _, info = env.step(action)
    record = Trajectory.load(env.iteration_dir).records()[0]
    context = record.get("env_params", decode_numbers=True)

    assert record["action"] == action
    assert context["nested"] == {"fixed": fixed_values}
    assert context["literal_list"] == ["keep", "both"]
    assert context["extra_env_vars"] == {"FIXED": "001"}
    assert context["extra_cmd_args"] == {"--precision": "bf16"}
    assert context["ball_speed"] == (info["env_params"]["ball_speed"] if randomized else 3)
    assert "paddle_width" not in context
    if node_action:
        assert "NUM_NODES" not in context
    else:
        assert context["NUM_NODES"] == 2
    assert initial_observation == [0.0]
    assert observation == [100.0] and reward == pytest.approx(0.01)
    if randomized:
        assert set(info["env_params"]) == set(initial_info["env_params"]) == {"ball_speed"}
        assert env.structured_observation_descriptors() == {"ball_speed": ObsLeafDescriptor(kind="discrete", n=2)}
    else:
        assert initial_info == {}
        assert "env_params" not in info
        assert info["trajectory"]["env_params"] == context
        assert env.structured_observation_descriptors() is None
    assert record.metadata["env_params"] == record["env_params"]


@pytest.fixture
def execution_env(
    tmp_path: Path, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    outcome, iteration = request.param
    tdef = TestDefinition(
        name="resolved",
        description="resolved",
        test_template_name="resolved",
        cmd_args=CmdArgs.model_validate(
            {"choice": [1, 2], "ball_speed": [1, 3], "fixed": {"label": "001", "enabled": True, "optional": None}}
        ),
        env_params={"ball_speed": EnvParamSpec(weights=[0.0, 1.0])},
        extra_env_vars={"FIXED": "001"},
        agent_metrics=["custom_metric", "bandwidth"],
        agent_reward_function="identity",
        agent_steps=2,
    )
    run = TestRun(
        name="resolved",
        test=tdef,
        num_nodes=2,
        nodes=[],
        current_iteration=iteration,
        time_limit="00:10:00",
        extra_srun_args="--exclusive",
    )
    slurm_system.global_env_vars["SAFE_SETTING"] = "launch-value"
    runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="resolved", test_runs=[run]),
        output_path=tmp_path / "run",
    )
    submitted: list[TestRun] = []
    measured: list[TestRun] = []

    def run_trial() -> None:
        submitted.append(runner.test_scenario.test_runs[0])
        if outcome == "runner_failure":
            raise RuntimeError("failed after receiving resolved run")
        submitted[-1].output_path.mkdir(parents=True, exist_ok=True)
        from cloudai.models.output import Metric, Run

        current = submitted[-1]
        runner.experiment_output.update_run(
            current.name,
            Run(
                path=str(current.output_path),
                jobid="job-173",
                iteration=current.current_iteration,
                step=current.step,
                status="failed" if outcome == "reported_runner_failure" else "completed",
                metrics=[
                    Metric(
                        name="custom_metric",
                        value=-1.0 if outcome in ("metric_failure", "legitimate_minus_one") else 5.0,
                        unit="unknown",
                    ),
                    Metric(name="bandwidth", value=2.5, unit="GB/s"),
                ],
            ),
        )
        if outcome == "completed_then_runner_failure":
            raise OSError("failed after publishing completed output")

    def metric(test_run: TestRun, system: System, name: str) -> MetricValue:
        del system
        measured.append(test_run)
        if name == "custom_metric":
            if outcome == "metric_failure":
                return METRIC_ERROR
            return -1.0 if outcome == "legitimate_minus_one" else 5.0
        return 2.5

    monkeypatch.setattr(runner, "run", run_trial)
    monkeypatch.setattr(TestRun, "get_metric_value", metric)
    return CloudAIGymEnv(run, runner, RewardOverrides()), submitted, measured, outcome, iteration


@pytest.mark.parametrize(
    "execution_env",
    [
        ("success", 0),
        ("metric_failure", 0),
        ("runner_failure", 0),
        ("legitimate_minus_one", 0),
        ("reported_runner_failure", 0),
    ],
    indirect=True,
)
def test_step_preserves_the_submitted_run_and_explicit_measurement_evidence(execution_env) -> None:
    env, submitted, measured, outcome, _ = execution_env
    initial, sampled_before = env.reset()
    observation, reward, done, info = env.step({"choice": 2})
    assert len(submitted) == 1
    assert env.test_run.test.cmd_args.choice == 2
    assert all(run.test.cmd_args.choice == 2 for run in measured)
    assert env.test_run.output_path == submitted[0].output_path
    assert initial == [0.0, 0.0] and not done
    assert sampled_before["env_params"] == info["env_params"] == {"ball_speed": 3}
    record = Trajectory.load(env.trajectory.output_path).records()[0]
    envelope = info["trajectory"]
    for domain in ("env_params", "configuration_id", "run_details", "measurement", "evaluation_failure"):
        assert envelope[domain] == record.get(domain, decode_numbers=True)
    assert envelope["env_params"]["fixed"] == {"label": "001", "enabled": True, "optional": None}
    assert envelope["env_params"]["ball_speed"] == 3
    assert "choice" not in envelope["env_params"]
    details = envelope["run_details"]
    assert (details["run_name"], details["iteration"], details["step"]) == ("resolved", 0, 1)
    assert details["output_path"] == str(submitted[0].output_path)
    assert details["history_id"]
    assert details["launch"]["num_nodes"] == 2
    assert details["launch"]["extra_srun_args"] == "--exclusive"
    assert details["launch"]["system"]["global_env_vars"]["SAFE_SETTING"] == "launch-value"
    measurement = envelope["measurement"]
    assert measurement["kind"] == "fresh"
    assert measurement["outcome"] == (
        "success"
        if outcome == "legitimate_minus_one"
        else "runner_failure"
        if outcome == "reported_runner_failure"
        else outcome
    )
    assert measurement["origin"]["history_id"] == details["history_id"]
    assert measurement["origin"]["step"] == details["step"]
    assert measurement["metrics"] == {
        "0": {
            "name": "custom_metric",
            "unit": "unknown",
            "status": "failed" if outcome == "metric_failure" else "observed",
        },
        "1": {"name": "bandwidth", "unit": "unknown" if outcome == "runner_failure" else "GB/s", "status": "observed"},
    }
    if outcome in ("success", "legitimate_minus_one"):
        assert envelope["evaluation_failure"] is None
        assert measurement["runner_error"] == ""
    else:
        assert envelope["evaluation_failure"] == {
            "constraint": False,
            "metric_indices": [0] if outcome == "metric_failure" else [],
            "runner": outcome in ("runner_failure", "reported_runner_failure"),
        }
        if outcome == "runner_failure":
            assert "failed after receiving resolved run" in measurement["runner_error"]
    assert observation == ([-1.0, 2.5] if outcome in ("metric_failure", "legitimate_minus_one") else [5.0, 2.5])
    assert reward == observation[0]


@pytest.mark.parametrize("execution_env", [("success", 2)], indirect=True)
def test_reset_preserves_the_requested_iteration_and_recording_location(execution_env) -> None:
    env, submitted, _, _, iteration = execution_env
    source = env.trajectory.output_path
    env.reset()
    assert env.test_run.current_iteration == iteration == 2
    env.step({"choice": 2})
    record = Trajectory.load(source).records()[0]
    assert record.source.parent == source.parent == env.iteration_dir
    assert submitted[0].current_iteration == 2
    assert record.get("run_details", decode_numbers=True)["iteration"] == 2


def test_cache_identity_includes_held_dimensions_and_preserves_original_measurement_origin(
    tmp_path: Path, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy

    from cloudai.models.output import Metric, Run

    definition = TestDefinition(
        name="cache",
        description="cache",
        test_template_name="cache",
        cmd_args=CmdArgs.model_validate({"x": [1, 2], "y": [10, 20], "fixed": "001"}),
        agent_metrics=["bandwidth"],
        agent_reward_function="identity",
        agent_steps=10,
    )
    template = TestRun(name="cache", test=definition, num_nodes=1, nodes=[])
    runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="cache", test_runs=[template]),
        output_path=tmp_path / "original",
    )
    submitted: list[TestRun] = []

    def execute() -> None:
        current = runner.test_scenario.test_runs[0]
        submitted.append(current)
        current.output_path.mkdir(parents=True, exist_ok=True)
        runner.experiment_output.update_run(
            current.name,
            Run(
                path=str(current.output_path),
                jobid=f"job-{current.step}",
                iteration=current.current_iteration,
                step=current.step,
                status="completed",
                metrics=[Metric(name="bandwidth", value=float(current.test.cmd_args.y), unit="GB/s")],
            ),
        )

    monkeypatch.setattr(runner, "run", execute)
    monkeypatch.setattr(TestRun, "get_metric_value", lambda run, system, metric: float(run.test.cmd_args.y))
    env = CloudAIGymEnv(template, runner, RewardOverrides())
    env.reset()
    proposals = ({"x": 1, "y": 10}, {"x": 2}, {"x": 1, "y": 20}, {"x": 2})
    results = [env.step(action) for action in proposals]
    observations = [result[0] for result in results]
    assert observations == [[10.0], [10.0], [20.0], [20.0]]
    assert len(submitted) == 4
    resolved_actions = [{"x": 1, "y": 10}, {"x": 2, "y": 10}, {"x": 1, "y": 20}, {"x": 2, "y": 20}]
    source_records = Trajectory.load(env.trajectory.output_path).records()
    for result, record, proposal, resolved_action in zip(
        results, source_records, proposals, resolved_actions, strict=True
    ):
        assert result[3]["trajectory"]["run_details"]["resolved_action"] == resolved_action
        assert record.get("run_details", decode_numbers=True)["resolved_action"] == resolved_action
        assert record["action"] == proposal
        assert record.get("env_params", decode_numbers=True)["fixed"] == "001"
    _, _, _, cached_info = env.step({"x": 2})
    assert cached_info["trajectory"]["run_details"]["resolved_action"] == {"x": 2, "y": 20}
    assert len(submitted) == 4
    assert cached_info["trajectory"]["measurement"]["kind"] == "cached"
    original_measurement = cached_info["trajectory"]["measurement"]["origin"]
    assert original_measurement["step"] == 4
    assert cached_info["trajectory"]["run_details"]["step"] == 5
    saved_run = copy.deepcopy(env.test_run)
    source = env.trajectory.output_path
    warm_frame = Trajectory.load(source).dataframe
    cached_only = warm_frame.loc[warm_frame["step"] == 5].copy()
    env.test_run.test.cmd_args.fixed = "changed"
    _, _, _, changed_info = env.step({"x": 2})
    assert changed_info["trajectory"]["run_details"]["resolved_action"] == {"x": 2, "y": 20}
    assert len(submitted) == 5
    assert changed_info["trajectory"]["measurement"]["kind"] == "fresh"
    assert changed_info["trajectory"]["configuration_id"] != cached_info["trajectory"]["configuration_id"]

    continued = TestRun(
        name="continued", test=copy.deepcopy(definition), num_nodes=1, nodes=[], current_iteration=2, step=5
    )
    continued_runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="continued", test_runs=[continued]),
        output_path=tmp_path / "continued",
    )
    continued_submitted: list[TestRun] = []

    def execute_continued() -> None:
        continued_submitted.append(continued_runner.test_scenario.test_runs[0])
        continued_submitted[-1].output_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(continued_runner, "run", execute_continued)
    monkeypatch.setattr(
        "cloudai.configurator.cloudai_gym.Trajectory",
        lambda *, iteration_dir: Trajectory(iteration_dir=iteration_dir, dataframe=cached_only),
    )
    resumed = CloudAIGymEnv(continued, continued_runner, RewardOverrides())
    saved_run.name = "continued"
    saved_run.current_iteration = 2
    resumed.test_run = saved_run
    resumed.reset()
    _, _, _, continued_info = resumed.step({"x": 2})
    assert continued_submitted == []
    continued_record = Trajectory.load(resumed.trajectory.output_path).records()[-1]
    current_details = continued_record.get("run_details", decode_numbers=True)
    assert current_details["resolved_action"] == {"x": 2, "y": 20}
    assert continued_record["action"] == {"x": 2}
    assert continued_info["trajectory"]["run_details"] == current_details
    measurement = continued_record.get("measurement", decode_numbers=True)
    assert (current_details["run_name"], current_details["iteration"], current_details["step"]) == ("continued", 2, 6)
    assert measurement["kind"] == "cached"
    assert measurement["origin"] == original_measurement
    assert current_details["history_id"] != original_measurement["history_id"]
    assert continued_info["trajectory"]["measurement"] == measurement

    from cloudai.core import BaseAgentConfig

    monkeypatch.setattr("cloudai.configurator.cloudai_gym.Trajectory", Trajectory)
    monkeypatch.setattr(TestRun, "get_metric_value", lambda run, system, metric: METRIC_ERROR)
    failure_run = TestRun(name="reward-first", test=copy.deepcopy(definition), num_nodes=1, nodes=[])
    failure_runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="reward-first", test_runs=[failure_run]),
        output_path=tmp_path / "reward-first",
    )

    def execute_failure() -> None:
        current = failure_runner.test_scenario.test_runs[0]
        current.output_path.mkdir(parents=True, exist_ok=True)
        failure_runner.experiment_output.update_run(
            current.name,
            Run(
                path=str(current.output_path),
                jobid="reward-first",
                iteration=current.current_iteration,
                step=current.step,
                status="completed",
                metrics=[],
            ),
        )

    monkeypatch.setattr(failure_runner, "run", execute_failure)
    first_rewards = BaseAgentConfig(rewards=RewardOverrides(metric_failure=-7.0)).rewards
    first_failure_env = CloudAIGymEnv(failure_run, failure_runner, first_rewards)
    _, first_reward, _, first_failure_info = first_failure_env.step({"x": 1, "y": 10})
    assert first_reward == -7.0
    failure_frame = Trajectory.load(first_failure_env.trajectory.output_path).dataframe
    different_run = TestRun(name="reward-second", test=copy.deepcopy(definition), num_nodes=1, nodes=[], step=1)
    different_runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="reward-second", test_runs=[different_run]),
        output_path=tmp_path / "reward-second",
    )
    different_submissions: list[TestRun] = []

    def execute_different() -> None:
        current = different_runner.test_scenario.test_runs[0]
        different_submissions.append(current)
        current.output_path.mkdir(parents=True, exist_ok=True)
        different_runner.experiment_output.update_run(
            current.name,
            Run(
                path=str(current.output_path),
                jobid="reward-second",
                iteration=current.current_iteration,
                step=current.step,
                status="completed",
                metrics=[],
            ),
        )

    monkeypatch.setattr(different_runner, "run", execute_different)
    monkeypatch.setattr(
        "cloudai.configurator.cloudai_gym.Trajectory",
        lambda *, iteration_dir: Trajectory(iteration_dir=iteration_dir, dataframe=failure_frame),
    )
    different_rewards = BaseAgentConfig(rewards=RewardOverrides(metric_failure=-17.0)).rewards
    different_env = CloudAIGymEnv(different_run, different_runner, different_rewards)
    _, different_reward, _, different_info = different_env.step({"x": 1, "y": 10})
    assert different_reward == -17.0
    assert len(different_submissions) == 1
    assert different_info["trajectory"]["measurement"]["kind"] == "fresh"
    assert different_info["trajectory"]["configuration_id"] != first_failure_info["trajectory"]["configuration_id"]


@pytest.mark.parametrize("relative_root", [False, True], ids=["absolute-output", "relative-output"])
def test_failed_status_from_the_public_runner_output_producer_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_root: bool
) -> None:
    from cloudai.core import JobStatusResult
    from cloudai.systems.standalone import StandaloneJob, StandaloneRunner, StandaloneSystem

    monkeypatch.chdir(tmp_path)
    output = Path("results") if relative_root else tmp_path / "results"
    definition = TestDefinition(
        name="status",
        description="status",
        test_template_name="status",
        cmd_args=CmdArgs.model_validate({"choice": [1, 2]}),
        agent_metrics=["default"],
        agent_reward_function="identity",
    )
    test_run = TestRun(name="status", test=definition, num_nodes=1, nodes=[])
    system = StandaloneSystem(name="status", install_path=tmp_path / "install", output_path=output)
    runner = StandaloneRunner(
        mode="dry-run",
        system=system,
        test_scenario=TestScenario(name="status", test_runs=[test_run]),
        output_path=output,
    )
    outputs = []

    def execute() -> None:
        current = runner.test_scenario.test_runs[0]
        current.output_path.mkdir(parents=True, exist_ok=True)
        job = StandaloneJob(test_run=current, id="reported-failure")
        result = runner.get_run_output(
            job, current, JobStatusResult(is_successful=False, error_message="reported failure")
        )
        outputs.append(result)
        runner.experiment_output.update_run(current.name, result)

    monkeypatch.setattr(runner, "run", execute)
    monkeypatch.setattr(TestRun, "get_metric_value", lambda run, system, metric: -1.0)
    env = CloudAIGymEnv(test_run, runner, RewardOverrides())
    observation, reward, _, info = env.step({"choice": 1})
    assert observation == [-1.0] and reward == -1.0
    assert outputs[0].status == "failed" and Path(outputs[0].path).is_absolute()
    measurement = info["trajectory"]["measurement"]
    assert measurement["execution_status"] == "failed"
    assert measurement["outcome"] == "runner_failure"
    assert measurement["origin"]["jobid"] == "reported-failure"
    assert info["trajectory"]["evaluation_failure"] == {"constraint": False, "metric_indices": [], "runner": True}


@pytest.mark.parametrize(
    "execution_env,expected_best",
    [
        (("success", 0), 1),
        (("legitimate_minus_one", 0), 1),
        (("metric_failure", 0), None),
        (("runner_failure", 0), None),
        (("reported_runner_failure", 0), None),
        (("completed_then_runner_failure", 0), None),
    ],
    indirect=["execution_env"],
)
def test_recommendations_use_measurement_eligibility_and_keep_completed_run_guard(execution_env, expected_best) -> None:
    env, submitted, _, outcome, _ = execution_env
    env.reset()
    env.step({"choice": 2})
    env.update_output()
    test_output = next(test for test in env.runner.experiment_output.experiment.tests if test.id == "resolved")
    assert test_output.dse is not None
    assert test_output.dse.best_step == expected_best
    records = Trajectory.load(env.trajectory.output_path).records()
    assert len(records) == 1
    measurements = [record for record in records if not record.ineligibility_reasons("measurement")]
    assert bool(measurements) is (expected_best is not None)
    assert not records[0].ineligibility_reasons("decision")
    if expected_best is not None:
        assert test_output.dse.best_config == {"choice": 2}
        env.step({"choice": 2})
        env.update_output()
        assert len(submitted) == 1
        records = Trajectory.load(env.trajectory.output_path).records()
        first = records[0].get("measurement", decode_numbers=True)
        cached = records[1].get("measurement", decode_numbers=True)
        assert cached["kind"] == "cached" and cached["origin"] == first["origin"]
        assert test_output.dse.best_step == 1
        assert sum(not record.ineligibility_reasons("measurement") for record in records) == 2
    elif outcome == "completed_then_runner_failure":
        assert test_output.runs[0].status == "completed"
        assert records[0].get("measurement", decode_numbers=True)["outcome"] == "runner_failure"


@pytest.mark.parametrize("execution_env", [("success", 0)], indirect=True)
@pytest.mark.parametrize(
    "proposal",
    ["partial", "complete", "legacy_complete", "unversioned_full", "unversioned_incomplete", "declared_incomplete"],
)
def test_recommendations_keep_resolved_held_dimensions_and_rank_order(
    execution_env: tuple[CloudAIGymEnv, list[TestRun], list[TestRun], str, int],
    monkeypatch: pytest.MonkeyPatch,
    proposal: str,
) -> None:
    import json
    from copy import deepcopy

    from cloudai.models.output import Metric, Run

    env, submitted, _, _, _ = execution_env
    env.original_test_run.test.cmd_args = CmdArgs.model_validate({"choice": [1, 2], "held": ["001", "002"]})
    env.test_run.test.cmd_args = deepcopy(env.original_test_run.test.cmd_args)
    monkeypatch.setattr(TestRun, "get_metric_value", lambda self, system, name: 9.0 if len(submitted) > 1 else 1.0)
    env.reset()
    first = {"choice": 1, "held": "001"}
    winning = {"choice": 2, "held": "001"}
    if proposal not in ("partial", "complete"):
        for step, action, reward in ((1, first, 1.0), (2, winning, 9.0)):
            context = {}
            if proposal in ("unversioned_full", "unversioned_incomplete"):
                context["run_details"] = {
                    "resolved_action": action if proposal == "unversioned_full" else {"choice": action["choice"]}
                }
            elif proposal == "declared_incomplete":
                seed = Trajectory(iteration_dir=env.trajectory.output_path.parent.parent / "seed" / str(step))
                selected = env.original_test_run.apply_params_set(action)
                selected.step = step
                native = seed.record(
                    test_run=selected,
                    action_space=env.original_test_run.param_space,
                    action=action,
                    reward=reward,
                    observation={"custom_metric": reward, "bandwidth": reward},
                    system=env.runner.system,
                    rewards=RewardOverrides(),
                    run_output=Run(
                        path=str(selected.output_path),
                        jobid=f"seed-{step}",
                        status="completed",
                        step=step,
                        iteration=0,
                        metrics=[
                            Metric(name="custom_metric", value=reward, unit="unknown"),
                            Metric(name="bandwidth", value=reward, unit="GB/s"),
                        ],
                    ),
                )
                details = native.get("run_details", decode_numbers=True)
                details["resolved_action"] = {"choice": action["choice"]}
                context.update(
                    record_context_version="1",
                    run_details=json.dumps(details),
                    measurement=native["measurement"],
                    evaluation_failure=native["evaluation_failure"],
                )
            env.trajectory.append(step=step, action=action, reward=reward, observation=[reward, reward], **context)
            env.runner.experiment_output.update_run(
                "resolved",
                Run(
                    path=str(env.trajectory.output_path.parent / str(step)),
                    jobid=f"legacy-{step}",
                    status="completed",
                    step=step,
                    iteration=0,
                    metrics=[Metric(name="custom_metric", value=reward, unit="unknown")],
                ),
            )
    else:
        env.step(first)
        env.step({"choice": 2} if proposal == "partial" else winning)
    env.update_output()
    output = next(test for test in env.runner.experiment_output.experiment.tests if test.id == "resolved")
    assert output.dse is not None
    assert output.dse.best_config is not None
    assert output.dse.best_config == winning
    assert output.dse.best_step == 2
    assert type(output.dse.best_config["held"]) is str
    records = Trajectory.load(env.trajectory.output_path).records()
    assert len(records) == 2
    if proposal in ("partial", "complete"):
        assert not any(record.ineligibility_reasons("measurement") for record in records)
        assert records[1].get("run_details", decode_numbers=True)["resolved_action"] == winning
        env.step(winning)
        env.update_output()
        assert len(submitted) == 2
        assert output.dse.best_step == 2 and output.dse.best_config == winning
        records = Trajectory.load(env.trajectory.output_path).records()
        assert (
            records[2].get("measurement", decode_numbers=True)["origin"]
            == records[1].get("measurement", decode_numbers=True)["origin"]
        )


@pytest.mark.parametrize("execution_env", [("success", 0)], indirect=True)
@pytest.mark.parametrize("archive", ["legacy", "native"])
@pytest.mark.parametrize(
    "rewards,expected_step",
    [
        pytest.param((9, "010"), 2, id="mixed_leading_zero"),
        pytest.param((9, " 10"), 2, id="mixed_whitespace"),
        pytest.param(("09", "010"), 2, id="lexical_strings"),
        pytest.param((9007199254740992, 9007199254740993), 2, id="native_bigints"),
        pytest.param((10, 10), 1, id="native_tie"),
        pytest.param((10, "010"), 1, id="mixed_numeric_tie"),
        pytest.param((9, " 10.5"), 2, id="whitespace_fraction"),
        pytest.param((9007199254740992, " 9007199254740993"), 2, id="mixed_bigint_string"),
    ],
)
def test_recommendations_rank_numeric_rewards_without_changing_recorded_values(
    execution_env: tuple[CloudAIGymEnv, list[TestRun], list[TestRun], str, int],
    archive: str,
    rewards: tuple[int | str, int | str],
    expected_step: int,
) -> None:
    from cloudai.models.output import Metric, Run

    env, _, _, _, _ = execution_env
    env.reset()
    writer = env.trajectory
    for step, reward in enumerate(rewards, 1):
        selected = env.original_test_run.apply_params_set({"choice": step}, env_params={"ball_speed": 3})
        selected.step = step
        selected.output_path = writer.output_path.parent / str(step)
        output = Run(
            path=str(selected.output_path),
            jobid=f"rank-{step}",
            status="completed",
            step=step,
            iteration=0,
            metrics=[
                Metric(name="custom_metric", value=1.0, unit="unknown"),
                Metric(name="bandwidth", value=2.5, unit="GB/s"),
            ],
        )
        if archive == "native":
            writer.record(
                test_run=selected,
                action_space=env.original_test_run.param_space,
                action={"choice": step},
                reward=reward,
                observation={"custom_metric": 1.0, "bandwidth": 2.5},
                system=env.runner.system,
                rewards=RewardOverrides(),
                run_output=output,
            )
        else:
            writer.append(step=step, action={"choice": step}, reward=reward, observation=[1.0, 2.5])
        env.runner.experiment_output.update_run("resolved", output)
    loaded = Trajectory.load(writer.output_path)
    before = {path: path.read_bytes() for path in writer.output_path.parent.iterdir() if path.is_file()}
    test_output = next(test for test in env.runner.experiment_output.experiment.tests if test.id == "resolved")
    outcomes = []
    for name, history in (("live", writer), ("loaded", loaded)):
        records = history.records()
        assert [record["reward"] for record in records] == list(rewards)
        assert [type(record["reward"]) for record in records] == [type(reward) for reward in rewards]
        assert not any(record.ineligibility_reasons("measurement") for record in records)
        getters = [(record.get("reward"), record.get("reward", decode_numbers=True)) for record in records]
        test_output.dse = None
        env.trajectory = history
        env.update_output()
        outcomes.append(
            (
                name,
                None if test_output.dse is None else test_output.dse.best_step,
                None if test_output.dse is None else test_output.dse.best_config,
            )
        )
        assert [
            (record.get("reward"), record.get("reward", decode_numbers=True)) for record in history.records()
        ] == getters
        assert [type(record["reward"]) for record in history.records()] == [type(reward) for reward in rewards]
    assert {path: path.read_bytes() for path in writer.output_path.parent.iterdir() if path.is_file()} == before
    assert outcomes == [
        ("live", expected_step, {"choice": expected_step}),
        ("loaded", expected_step, {"choice": expected_step}),
    ]
