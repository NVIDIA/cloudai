# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cloudai.configurator import CloudAIGymEnv, GridSearchAgent, TrajectoryEntry
from cloudai.core import BaseRunner, RewardOverrides, Runner, TestRun, TestScenario
from cloudai.systems.slurm import SlurmSystem
from cloudai.util import flatten_dict
from cloudai.workloads.nemo_run import (
    Data,
    NeMoRunCmdArgs,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)
from cloudai.workloads.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition


@pytest.fixture
def nemorun() -> NeMoRunTestDefinition:
    return NeMoRunTestDefinition(
        name="NemoModel",
        description="Nemo Model",
        test_template_name="nemo_template",
        cmd_args=NeMoRunCmdArgs(docker_image_url="https://docker/url", task="some_task", recipe_name="some_recipe"),
    )


@pytest.fixture
def setup_env(slurm_system: SlurmSystem, nemorun: NeMoRunTestDefinition) -> tuple[TestRun, BaseRunner]:
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000],
        val_check_interval=[100, 200],
        strategy=TrainerStrategy(
            tensor_model_parallel_size=[1, 2],
            pipeline_model_parallel_size=[1, 2],
            context_parallel_size=[2, 4],
        ),
    )
    tdef.cmd_args.data = Data(micro_batch_size=[1, 2])
    tdef.agent_metrics = ["default"]

    mock_command_gen = MagicMock()
    mock_command_gen.gen_srun_command.return_value = "srun mock command"
    mock_command_gen.generate_test_command.return_value = ["python", "run.py", "--arg", "value"]

    test_template_mock = MagicMock()
    test_template_mock.command_gen_strategy = mock_command_gen

    test_run = TestRun(
        name="mock_test_run", test=tdef, num_nodes=1, nodes=[], reports={NeMoRunReportGenerationStrategy}
    )

    test_scenario = TestScenario(name="mock_test_scenario", test_runs=[test_run])
    test_run.output_path = (
        slurm_system.output_path / test_scenario.name / test_run.name / f"{test_run.current_iteration}"
    )

    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    return test_run, runner.runner


def test_observation_space(setup_env: tuple[TestRun, BaseRunner]):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    observation_space = env.define_observation_space()

    expected_observation_space = [0.0]

    assert observation_space == expected_observation_space


@pytest.mark.parametrize(
    "reward_function,test_cases",
    [
        (
            "inverse",
            [
                ([0.34827126874999986], pytest.approx(2.871, 0.001)),
                ([0.0], 0.0),
                ([], 0.0),
                ([2.0, 2.0], 0.5),
            ],
        ),
        (
            "negative",
            [
                ([2.0], -2.0),
                ([-1.5], 1.5),
                ([0.0], 0.0),
                ([], 0.0),
            ],
        ),
        (
            "identity",
            [
                ([2.0], 2.0),
                ([-1.5], -1.5),
                ([0.0], 0.0),
                ([], 0.0),
            ],
        ),
    ],
)
def test_compute_reward(reward_function, test_cases, base_tr: TestRun):
    base_tr.test.agent_reward_function = reward_function
    env = CloudAIGymEnv(test_run=base_tr, runner=MagicMock(), rewards=RewardOverrides())

    for input_value, expected_reward in test_cases:
        reward = env.compute_reward(input_value)
        assert reward == expected_reward


def test_compute_reward_invalid(base_tr: TestRun):
    base_tr.test.agent_reward_function = "nonexistent"

    with pytest.raises(KeyError) as exc_info:
        CloudAIGymEnv(test_run=base_tr, runner=MagicMock(), rewards=RewardOverrides())

    assert "Reward function 'nonexistent' not found" in str(exc_info.value)
    assert (
        "Available functions: ['inverse', 'negative', 'identity', "
        "'ai_dynamo_weighted_normalized', 'ai_dynamo_ratio_normalized', 'ai_dynamo_log_scale']" in str(exc_info.value)
    )


def test_tr_output_path(setup_env: tuple[TestRun, BaseRunner]):
    test_run, runner = setup_env
    test_run.test.cmd_args.data.global_batch_size = 8  # avoid constraint check failure
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    agent = GridSearchAgent(env, GridSearchAgent.get_config_class()())

    _, action = agent.select_action()
    env.test_run.step = 41
    env.step(action)

    assert env.test_run.output_path.name == "42", (
        "CloudAIGymEnv.step() must advance test_run.step before computing output_path; "
        "starting at 41, step #42's artifacts must land in dir '42'."
    )


@pytest.mark.parametrize(
    "rewards, expected_reward",
    [
        pytest.param(RewardOverrides(), -1.0, id="default_penalty"),
        pytest.param(RewardOverrides(constraint_failure=-2.5), -2.5, id="custom_penalty"),
    ],
)
def test_constraint_failure(nemorun: NeMoRunTestDefinition, rewards: RewardOverrides, expected_reward: float):
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="constraint_fail_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.system = MagicMock()
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=rewards)

    bad = {"trainer.strategy.context_parallel_size": 3}  # induce constraint failure
    obs, reward, done, info = env.step(bad)

    assert obs == [-1.0]
    assert reward == expected_reward
    assert done is True
    assert info == {}


def test_action_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2])
    )
    nemorun.cmd_args.data.micro_batch_size = [1, 2]
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2"]

    tr.test = nemorun
    tr.num_nodes = [1, 2]

    action_space = tr.param_space

    assert len(action_space) == 5
    assert action_space["data.micro_batch_size"] == nemorun.cmd_args.data.micro_batch_size
    assert action_space["trainer.max_steps"] == nemorun.cmd_args.trainer.max_steps
    assert (
        action_space["trainer.strategy.tensor_model_parallel_size"]
        == nemorun.cmd_args.trainer.strategy.tensor_model_parallel_size
    )
    assert action_space["extra_env_vars.DSE_VAR"] == nemorun.extra_env_vars["DSE_VAR"]
    assert action_space["NUM_NODES"] == tr.num_nodes


def test_action_space_excludes_configured_cmd_arg_prefix(
    nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]
):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2])
    )
    nemorun.dse_excluded_args = ["cmd_args.trainer.strategy"]
    tr.test = nemorun

    action_space = tr.param_space

    assert action_space["trainer.max_steps"] == [1000, 2000]
    assert "trainer.strategy.tensor_model_parallel_size" not in action_space


@pytest.mark.parametrize("num_nodes", (1, [1, 2], [3]))
def test_all_combinations(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner], num_nodes: int):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2]))
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2", "3"]
    tr.test = nemorun
    tr.num_nodes = num_nodes

    expected_num_combinations = 6
    if isinstance(num_nodes, list):
        expected_num_combinations *= len(num_nodes)

    real_combinations = tr.all_combinations
    assert len(real_combinations) == expected_num_combinations
    _combinations = [
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "3"},
    ]
    expected_combinations = []
    for param_set in _combinations:
        if isinstance(num_nodes, list):
            for nnodes in num_nodes:
                expected_combinations.append(param_set | {"NUM_NODES": nnodes})
        else:
            expected_combinations.append(param_set)

    for expected in expected_combinations:
        assert expected in real_combinations, f"Expected {expected} in all_combinations"


def test_all_combinations_non_dse(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]):
    tr, _ = setup_env
    tr.test = nemorun
    assert len(tr.all_combinations) == 0


def test_all_combinations_non_dse_but_with_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test = nemorun
    with patch.object(type(tr.test), "is_dse_job", new_callable=PropertyMock(return_value=True)):
        assert len(tr.all_combinations) == 0


def test_all_combinations_dse_on_num_nodes(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test = NeMoRunTestDefinition(
        name="NemoModel",
        description="Nemo Model",
        test_template_name="nemo_template",
        cmd_args=NeMoRunCmdArgs(docker_image_url="https://docker/url", task="some_task", recipe_name="some_recipe"),
    )
    tr.num_nodes = [1, 2]
    assert len(tr.all_combinations) == 2


@pytest.mark.parametrize("num_nodes", (1, [1, 2], [3]))
def test_params_set(setup_env: tuple[TestRun, Runner], num_nodes: int):
    tr, _ = setup_env
    tr.num_nodes = num_nodes
    assert len(tr.all_combinations) > 1
    for action in tr.all_combinations:
        new_tr = tr.apply_params_set(action)
        cmd_args = flatten_dict(new_tr.test.cmd_args.model_dump())
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                assert new_tr.test.extra_env_vars[key[len("extra_env_vars.") :]] == value
            elif key == "NUM_NODES":
                assert new_tr.num_nodes == value
            else:
                assert cmd_args[key] == value


def test_params_set_validated(setup_env: tuple[TestRun, Runner], nemorun: NeMoRunTestDefinition):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000])
    tr.test = nemorun
    action_space = tr.param_space
    action_space["trainer.max_steps"] = "invalid"

    with pytest.raises(UserWarning) as excinfo:
        tr.apply_params_set(action_space)

    assert excinfo.type is UserWarning
    assert "Pydantic serializer warnings:" in str(excinfo.value)
    assert "serialized value may not be as expected" in str(excinfo.value)
    assert "input_value='invalid'" in str(excinfo.value)


def test_apply_params_set__preserves_installables_state(setup_env: tuple[TestRun, Runner], tmp_path: Path):
    tr, _ = setup_env
    tr.test = NIXLBenchTestDefinition(
        name="NIXLBench",
        description="NIXL Bench",
        test_template_name="NIXLBench",
        cmd_args=NIXLBenchCmdArgs(
            docker_image_url="https://docker/url",
            path_to_benchmark="https://benchmark/path",
        ),
    )
    tr.test.docker_image.installed_path = tmp_path

    new_tr = tr.apply_params_set({"backend": "VRAM"})

    upd_tdef = cast(NIXLBenchTestDefinition, new_tr.test)

    assert upd_tdef.docker_image.installed_path == tmp_path


@pytest.mark.parametrize(
    ("trajectory", "current_iteration", "action", "expected_step"),
    [
        ({}, 0, {"x": 1}, None),
        ({0: [TrajectoryEntry(1, {"x": 1}, 1, [1])]}, 0, {"x": 1}, 1),
        ({0: [TrajectoryEntry(1, {"x": 1.0}, 1, [1])]}, 0, {"x": 1}, None),
        (
            {
                0: [
                    TrajectoryEntry(1, {"x": 1.0}, 1, [1]),
                    TrajectoryEntry(2, {"x": 1}, 1, [1]),
                ]
            },
            0,
            {"x": 1},
            2,
        ),
        ({0: [TrajectoryEntry(1, {"x": 1}, 1, [1])]}, 1, {"x": 1}, None),
        ({1: [TrajectoryEntry(3, {"x": 1}, 1, [1])]}, 1, {"x": 1}, 3),
    ],
)
def test_get_cached_trajectory_result(
    base_tr: TestRun,
    tmp_path: Path,
    trajectory: dict[int, list[TrajectoryEntry]],
    current_iteration: int,
    action: dict[str, object],
    expected_step: int | None,
) -> None:
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}
    runner.get_job_output_path.return_value = tmp_path / "scenario" / base_tr.name / "0" / "7"

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    env.test_run.current_iteration = current_iteration
    env.trajectory = trajectory

    actual = env.get_cached_trajectory_result(action)
    if actual is None:
        assert expected_step is None
    else:
        assert actual.step == expected_step


def test_cached_step_appends_trajectory_row(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Cache hits must still append a row to trajectory.csv so the visible step list matches agent_steps."""
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="cache_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    cached_action = {"trainer.max_steps": 1000}
    env.test_run.current_iteration = 0
    env.trajectory = {0: [TrajectoryEntry(step=1, action=cached_action, reward=0.42, observation=[0.84])]}

    env.test_run.step = 4
    obs, reward, done, _info = env.step(cached_action)

    runner.run.assert_not_called()
    assert reward == 0.42
    assert obs == [0.84]
    assert done is False
    rows = env.trajectory[0]
    assert len(rows) == 2
    assert rows[-1].step == 5, (
        "CloudAIGymEnv.step() advances test_run.step before recording the trajectory row; "
        "the cached row must be tagged with the advanced trial index, not the pre-step value."
    )
    assert rows[-1].reward == 0.42
    assert rows[-1].action == cached_action

    csv_path = env.trajectory_file_path
    assert csv_path.exists()
    contents = csv_path.read_text().strip().splitlines()
    assert contents[0] == "step,action,reward,observation"
    assert contents[-1].startswith("5,")


def _seed_cached_entry_with_env_params(
    env: CloudAIGymEnv, action: dict[str, object], env_params: dict[str, object]
) -> None:
    """Seed env.trajectory with one entry, attaching env_params via object.__setattr__.

    TrajectoryEntry is a frozen dataclass and does not yet declare env_params.
    Once the field is added, drop this helper and pass env_params as a kwarg.
    """
    entry = TrajectoryEntry(step=1, action=action, reward=0.5, observation=[100.0])
    object.__setattr__(entry, "env_params", env_params)
    env.test_run.current_iteration = 0
    env.trajectory = {0: [entry]}


def test_cache_miss_when_env_params_differ(base_tr: TestRun, tmp_path: Path) -> None:
    """Cache MUST miss when env_params differ, even if action is identical.

    Without this property the agent receives stale rewards on every cache hit
    under domain randomization. PPO/DQN/BO all silently train on labels that
    do not correspond to the env they were nominally generated under.
    """
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"drop_rate": 0.001})

    env.test_run.current_env_params = {"drop_rate": 0.01}  # type: ignore[attr-defined]

    assert env.get_cached_trajectory_result({"x": 10}) is None, (
        "Cache must include env_params in its key. The current implementation "
        "keys on action alone, so trials repeating the same action under a "
        "different env_params sample receive a stale cached reward. See "
        "env-params-cloudai-corpus-plan.md."
    )


def test_cache_hit_when_action_and_env_params_match(base_tr: TestRun, tmp_path: Path) -> None:
    """Same action AND same env_params must still HIT the cache."""
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"drop_rate": 0.001})

    env.test_run.current_env_params = {"drop_rate": 0.001}  # type: ignore[attr-defined]

    result = env.get_cached_trajectory_result({"x": 10})
    assert result is not None
    assert result.step == 1


def test_cache_hit_when_neither_has_env_params(base_tr: TestRun, tmp_path: Path) -> None:
    """Workloads without env_params behave exactly as today (back-compat)."""
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    env.test_run.current_iteration = 0
    env.trajectory = {0: [TrajectoryEntry(step=1, action={"x": 10}, reward=0.5, observation=[100.0])]}
    # Note: neither the cached entry nor test_run carries env_params -> existing behavior.

    result = env.get_cached_trajectory_result({"x": 10})
    assert result is not None
    assert result.step == 1


def test_step_reruns_workload_when_env_params_change(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Integration: env.step() with same action but different env_params re-runs the workload.

    Counterpart to test_cache_miss_when_env_params_differ but exercising the
    full step() flow: increment_step -> apply_params_set -> cache lookup ->
    runner.run() -> write_trajectory.
    """
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = test_scenario
    runner.jobs = {}
    runner.testrun_to_job_map = {}
    runner.shutting_down = False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    action = {"trainer.max_steps": 1000}
    fake_obs = iter([[100.0], [50.0]])

    with patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)):
        env.test_run.step = 0
        env.test_run.current_env_params = {"drop_rate": 0.001}  # type: ignore[attr-defined]
        obs1, _r1, *_ = env.step(action)

        env.test_run.current_env_params = {"drop_rate": 0.01}  # type: ignore[attr-defined]
        obs2, _r2, *_ = env.step(action)

    assert runner.run.call_count == 2, (
        "Different env_params between two env.step() calls with the same action "
        "must trigger a workload re-run; the cache lookup must miss."
    )
    assert obs1 != obs2, "fresh workload run should produce a fresh observation"
