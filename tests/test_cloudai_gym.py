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


def test_cached_numeric_text_returns_runtime_float_without_rewriting_recorded_reward(
    slurm_system: SlurmSystem, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = TestRun(
        name="cached-numeric-text",
        test=TestDefinition(
            name="cache",
            description="cache reward wire",
            test_template_name="cache",
            cmd_args=CmdArgs.model_validate({"x": [1, 2]}),
            agent_metrics=["default"],
        ),
        num_nodes=1,
        nodes=[],
        step=1,
    )
    runner = SlurmRunner(
        mode="dry-run",
        system=slurm_system,
        test_scenario=TestScenario(name="cache", test_runs=[template]),
        output_path=tmp_path,
    )
    env = CloudAIGymEnv(template, runner, RewardOverrides())
    cached = env.original_test_run.apply_params_set({"x": 1})
    cached.step = 1
    cached.output_path = env.iteration_dir / "1"
    env.trajectory.record(
        test_run=cached,
        action_space=template.param_space,
        action={"x": 1},
        reward="010",
        observation={"default": 1.0},
        system=slurm_system,
        rewards=env.rewards,
    )

    def reject_execution() -> None:
        pytest.fail("an exact cache hit must not execute the runner")

    monkeypatch.setattr(runner, "run", reject_execution)
    before = {path: path.read_bytes() for path in (env.trajectory.output_path, env.trajectory.metadata_path)}
    assert env.get_cached_trajectory_result({"x": 1}, {}) is not None

    observation, reward, _, info = env.step({"x": 1})

    assert observation == [1.0]
    assert type(reward) is float and reward == 10.0
    assert info["trajectory"]["measurement"]["kind"] == "cached"
    records = Trajectory.load(env.iteration_dir).records()
    assert [record["reward"] for record in records] == ["010", 10.0]
    assert type(records[0]["reward"]) is str and type(records[1]["reward"]) is float
    assert all(path.read_bytes().startswith(content) for path, content in before.items())
    assert (
        records[1].get("measurement", decode_numbers=True)["origin"]
        == records[0].get("measurement", decode_numbers=True)["origin"]
    )
    warm = env.trajectory.dataframe
    warm["observation.default"] = [1.0] * len(warm)
    warm = warm.drop(columns=["observation"]).astype(object)
    env.trajectory = Trajectory(iteration_dir=tmp_path / "warm", dataframe=warm)
    observation, reward, _, info = env.step({"x": 1})
    assert observation == [1.0] and type(reward) is float and reward == 10.0
    assert info["trajectory"]["measurement"]["kind"] == "cached"
