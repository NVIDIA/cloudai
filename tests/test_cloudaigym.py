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
from cloudai.configurator.cloudai_gym import _create_gym_server
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import BaseRunner, CmdArgs, RewardOverrides, Runner, TestDefinition, TestRun, TestScenario
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
from tests.test_env_params import EnvVarCmdArgs, EnvVarTestDefinition


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
    """Seed env.trajectory with one entry carrying the given env_params."""
    entry = TrajectoryEntry(step=1, action=action, reward=0.5, observation=[100.0], env_params=env_params)
    env.test_run.current_iteration = 0
    env.trajectory = {0: [entry]}


def test_cache_miss_when_env_params_differ(base_tr: TestRun, tmp_path: Path) -> None:
    """Cache MUST miss when env_params differ, even if action is identical.

    Without this property the agent receives stale rewards on every cache hit
    when the env varies per trial: any agent silently trains on labels that
    do not correspond to the env they were nominally generated under.
    """
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"ball_speed": 1})

    env.test_run.current_env_params = {"ball_speed": 2}

    assert env.get_cached_trajectory_result({"x": 10}) is None, (
        "Cache must include env_params in its key. Keying on action alone means "
        "trials repeating the same action under a different env_params sample "
        "receive a stale cached reward."
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
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"ball_speed": 2})

    env.test_run.current_env_params = {"ball_speed": 2}

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


def test_step_reruns_workload_when_env_params_change(tmp_path: Path) -> None:
    """Integration: two env.step() calls with the same action but different sampled env_params re-run.

    Counterpart to test_cache_miss_when_env_params_differ but exercising the
    full step() flow: increment_step -> sample env_params -> apply_params_set ->
    cache lookup -> runner.run() -> write_trajectory. With seed 42 the sampler
    draws ball_speed=3 then ball_speed=1 on the two consecutive trials, so the
    cache key differs and the workload must re-run both times.
    """
    tdef = EnvVarTestDefinition(
        name="dr",
        description="dr",
        test_template_name="dr_template",
        cmd_args=EnvVarCmdArgs(ball_speed=[1, 2, 3]),
        env_params={"ball_speed": EnvParamSpec()},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )
    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
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
    action = {"paddle_width": 4}
    fake_obs = iter([[100.0], [50.0]])

    with patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)):
        env.test_run.step = 0
        obs1, _r1, *_ = env.step(action)  # samples ball_speed=3
        obs2, _r2, *_ = env.step(action)  # samples ball_speed=1

    assert runner.run.call_count == 2, (
        "Different sampled env_params between two env.step() calls with the same action "
        "must trigger a workload re-run; the cache lookup must miss."
    )
    assert obs1 != obs2, "fresh workload run should produce a fresh observation"


def test_env_csv_is_step_aligned_with_trajectory(tmp_path: Path) -> None:
    """env.csv must have exactly one row per env.step() call, with steps aligned 1:1 to trajectory.csv.

    This pins the corpus-friendly contract: a downstream consumer can
    ``pd.merge(traj, env, on="step")`` without losing rows on either side,
    independent of whether the trial hit the trajectory cache.
    """
    tdef = EnvVarTestDefinition(
        name="dr",
        description="dr",
        test_template_name="dr_template",
        cmd_args=EnvVarCmdArgs(ball_speed=[1, 2, 3]),
        env_params={"ball_speed": EnvParamSpec()},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )

    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
    )
    test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = test_scenario
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    action_a, action_b = {"paddle_width": 4}, {"paddle_width": 8}
    fake_obs = iter([[100.0], [50.0], [25.0]])

    with patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)):
        env.test_run.step = 0
        for action in (action_a, action_b, action_a):
            env.step(action)

    env_csv = env.env_params_record_path
    traj_csv = env.trajectory_file_path
    assert env_csv.exists(), "env.csv must be written when env_params is declared"

    env_steps = [int(line.split(",", 1)[0]) for line in env_csv.read_text().strip().splitlines()[1:]]
    traj_steps = [int(line.split(",", 1)[0]) for line in traj_csv.read_text().strip().splitlines()[1:]]
    assert env_steps == traj_steps == [1, 2, 3], (
        f"step columns must align 1:1 across env.csv ({env_steps}) and trajectory.csv ({traj_steps})"
    )


def test_env_csv_step_alignment_holds_on_constraint_failure(tmp_path: Path) -> None:
    """A constraint failure must not desync env.csv from trajectory.csv.

    Runs three steps where the middle one fails ``constraint_check`` and the
    other two succeed. ``env.csv`` is sunk inside ``write_trajectory`` from the
    same ``TrajectoryEntry``, which is never reached on the early-return
    constraint-failure path - so the failed step lands in neither file. The
    corpus-friendly contract (``pd.merge(traj, env, on="step")`` loses no rows)
    therefore holds via shared absence: both files record exactly the surviving
    steps, aligned 1:1.
    """
    tdef = EnvVarTestDefinition(
        name="dr",
        description="dr",
        test_template_name="dr_template",
        cmd_args=EnvVarCmdArgs(ball_speed=[1, 2, 3]),
        env_params={"ball_speed": EnvParamSpec()},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )

    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
    )
    test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = test_scenario
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())

    # Step 2 fails the constraint; steps 1 and 3 survive. get_observation is only
    # reached on the surviving steps, so it yields exactly two values.
    fake_obs = iter([[100.0], [25.0]])
    with (
        patch.object(EnvVarTestDefinition, "constraint_check", side_effect=[True, False, True]),
        patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)),
    ):
        env.test_run.step = 0
        for action in ({"paddle_width": 4}, {"paddle_width": 6}, {"paddle_width": 8}):
            env.step(action)

    env_csv = env.env_params_record_path
    traj_csv = env.trajectory_file_path

    assert env_csv.exists(), "surviving steps declare env_params -> env.csv must exist"
    env_steps = [int(line.split(",", 1)[0]) for line in env_csv.read_text().strip().splitlines()[1:]]
    traj_steps = (
        [int(line.split(",", 1)[0]) for line in traj_csv.read_text().strip().splitlines()[1:]]
        if traj_csv.exists()
        else []
    )
    assert env_steps == traj_steps == [1, 3], (
        f"the constraint-failed step (2) must appear in neither file; env.csv ({env_steps}) "
        f"and trajectory.csv ({traj_steps}) must stay 1:1 aligned on the surviving steps"
    )


def test_step_cache_hit_with_declared_env_params_still_writes_env_csv(tmp_path: Path) -> None:
    """End-to-end: cache HIT under observer-driven env_params still records env.csv.

    A cache hit still calls ``write_trajectory``, which sinks the trajectory row
    and the matching env.csv row from the same entry - keeping the two files
    step-aligned even when the workload itself is short-circuited.
    Asserts: (a) the workload is NOT re-run (cache short-circuit), (b)
    env.csv gains a row, (c) trajectory.csv gains a row carrying the
    sampled env_params.
    """
    import random as _random

    tdef = EnvVarTestDefinition(
        name="dr",
        description="dr",
        test_template_name="dr_template",
        cmd_args=EnvVarCmdArgs(ball_speed=[1, 2, 3]),
        env_params={"ball_speed": EnvParamSpec()},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )

    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    assert env.params is not None, "TestDefinition.env_params declared -> EnvParams must be built"

    expected_sample = {"ball_speed": _random.Random("42:ball_speed:1").choice([1, 2, 3])}
    action = {"paddle_width": 4}
    env.test_run.current_iteration = 0
    env.trajectory = {
        0: [TrajectoryEntry(step=0, action=action, reward=0.42, observation=[0.84], env_params=expected_sample)]
    }
    env.test_run.step = 0

    with patch.object(env, "get_observation", side_effect=AssertionError("cache miss path must not run")):
        obs, reward, _done, _info = env.step(action)

    runner.run.assert_not_called()
    assert reward == 0.42 and obs == [0.84]

    env_csv = env.env_params_record_path
    assert env_csv.exists(), "cache HIT must NOT skip the observer; env.csv must record the trial"
    env_rows = env_csv.read_text().strip().splitlines()
    assert env_rows[0] == "step,env"
    assert env_rows[1].startswith("1,"), f"expected step 1 row in env.csv, got {env_rows[1]!r}"

    traj_rows = env.trajectory[0]
    assert len(traj_rows) == 2 and traj_rows[-1].env_params == expected_sample, (
        "cache-hit trajectory entry must record the per-trial env_params sample"
    )


def test_step_overlays_env_params_onto_cmd_args(tmp_path: Path) -> None:
    """The per-trial env_params sample must be overlaid onto cmd_args before the workload runs.

    env_params are the env-side twin of the action: core overlays the sampled values onto
    cmd_args inside step() so the workload actually runs with them, workload-agnostically
    (no per-workload injection code). Here ``ball_speed`` is env-randomized (its candidate
    list lives in cmd_args), so the value the workload runs with must equal the observer's
    sample, not the whole candidate list.
    """
    import random as _random

    tdef = EnvVarTestDefinition(
        name="overlay",
        description="overlay",
        test_template_name="dr_template",
        cmd_args=EnvVarCmdArgs(ball_speed=[1, 2]),
        env_params={"ball_speed": EnvParamSpec()},
        agent_metrics=["default"],
        agent_config={"random_seed": 42},
    )

    test_run = TestRun(
        name="overlay_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "overlay_tr" / "0",
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="overlay_scenario", test_runs=[test_run])
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    expected = _random.Random("42:ball_speed:1").choice([1, 2])

    with patch.object(env, "get_observation", side_effect=lambda _action: [1.0]):
        env.test_run.step = 0
        env.step({"paddle_width": 4})

    ran_cmd_args = runner.test_scenario.test_runs[0].test.cmd_args
    assert ran_cmd_args.ball_speed == expected, (
        "step() must overlay the sampled env_params onto cmd_args before runner.run(), so the "
        f"workload runs with ball_speed={expected} (the observer's draw), not the candidate list."
    )


def test_param_space_excludes_env_params_keys(setup_env: tuple[TestRun, BaseRunner]):
    """env_params keys must never surface in the grid/action space (sampled, not searched)."""
    tr, _ = setup_env
    tr.test = EnvVarTestDefinition(
        name="dr",
        description="dr",
        test_template_name="dr_template",
        # paddle_width is an ordinary action-space list; ball_speed is an env-sampled list.
        cmd_args=EnvVarCmdArgs(paddle_width=[4, 8], ball_speed=[1, 2]),
        env_params={"ball_speed": EnvParamSpec()},
    )

    action_space = tr.param_space

    assert "paddle_width" in action_space, "an un-annotated cmd_args list must remain an action-space dimension"
    assert "ball_speed" not in action_space, (
        "a knob declared in env_params is sampled by the env, not explored by the agent, so it "
        "must be excluded from param_space even though its cmd_args value is a list."
    )


def test_no_env_csv_when_env_params_not_declared(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Workloads without [env_params.*] pay zero overhead: no observer, no env.csv."""
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    test_run = TestRun(
        name="plain_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "plain_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())

    assert env.params is None, "no env_params declared -> no EnvParams object"
    assert not env.env_params_record_path.exists()


class FakeGymServer:
    """In-process GymServer stub for online (live-RL) mode tests."""

    def __init__(self, n_actions: int = 2):
        self.n_actions = n_actions
        self.reset_calls = 0
        self.step_calls = 0
        self.last_action: object = None

    def reset(self) -> tuple[list[float], dict[str, object]]:
        self.reset_calls += 1
        return [0.0] * self.n_actions, {"reset": True}

    def step(self, action: dict[str, object]) -> tuple[list[float], float, bool, dict[str, object]]:
        self.step_calls += 1
        self.last_action = action
        return [1.0, 2.0], 3.5, False, {"served": True}

    def get_action_space(self) -> dict[str, object]:
        return {"a": [0, 1], "b": [0, 1]}

    def get_observation_space(self) -> list[float]:
        return [0.0, 0.0]


def test_default_mode_is_offline(base_tr: TestRun) -> None:
    """Workloads without live_rl_mode keep the runner-backed (offline) behavior."""
    env = CloudAIGymEnv(test_run=base_tr, runner=MagicMock(spec=BaseRunner), rewards=RewardOverrides())
    assert env._is_online is False
    assert env._gym_server is None


def test_online_mode_delegates_to_gym_server(base_tr: TestRun, tmp_path: Path) -> None:
    """An injected GymServer drives action/observation spaces, reset, and step without the runner."""
    base_tr.output_path = tmp_path / "out"
    server = FakeGymServer(n_actions=2)
    runner = MagicMock(spec=BaseRunner)

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides(), gym_server=server)

    assert env._is_online is True
    assert env.define_action_space() == {"a": [0, 1], "b": [0, 1]}
    assert env.define_observation_space() == [0.0, 0.0]

    obs, info = env.reset()
    assert obs == [0.0, 0.0]
    assert info == {"reset": True}

    obs, reward, done, info = env.step({"a": 1})
    assert obs == [1.0, 2.0]
    assert reward == 3.5
    assert done is False
    assert info == {"served": True}

    runner.run.assert_not_called()
    assert server.step_calls == 1
    assert server.last_action == {"a": 1}


def test_online_step_writes_trajectory_to_output_path(base_tr: TestRun, tmp_path: Path) -> None:
    """Online steps still produce trajectory.csv (under output_path), counted via test_run.step."""
    base_tr.output_path = tmp_path / "out"
    env = CloudAIGymEnv(
        test_run=base_tr, runner=MagicMock(spec=BaseRunner), rewards=RewardOverrides(), gym_server=FakeGymServer()
    )

    env.reset()
    env.step({"a": 1})
    env.step({"a": 0})

    traj = env.trajectory_file_path
    assert traj == base_tr.output_path / "trajectory.csv"
    lines = traj.read_text().strip().splitlines()
    assert lines[0] == "step,action,reward,observation"
    assert lines[1].startswith("1,")
    assert lines[2].startswith("2,")


def test_online_step_numbering_is_monotonic_across_resets(base_tr: TestRun, tmp_path: Path) -> None:
    """reset() must not rewind online step numbering: under reset-per-episode rollouts the
    trajectory rows keep increasing (via monotonic test_run.step) instead of collapsing to 1.
    """
    base_tr.output_path = tmp_path / "out"
    env = CloudAIGymEnv(
        test_run=base_tr, runner=MagicMock(spec=BaseRunner), rewards=RewardOverrides(), gym_server=FakeGymServer()
    )

    env.reset()
    env.step({"a": 1})
    env.step({"a": 0})
    env.reset()
    env.step({"a": 1})

    lines = env.trajectory_file_path.read_text().strip().splitlines()
    assert [line.split(",", 1)[0] for line in lines[1:]] == ["1", "2", "3"]


def test_create_gym_server_imports_and_filters_kwargs() -> None:
    """env_class is imported and cmd_args are forwarded, filtered to the server's __init__ signature."""
    tr = MagicMock()
    tr.test.cmd_args.model_dump.return_value = {
        "env_class": f"{FakeGymServer.__module__}.FakeGymServer",
        "live_rl_mode": True,
        "docker_image_url": "https://docker/url",
        "n_actions": 3,
        "unknown_param": 99,
    }

    server = _create_gym_server(tr)

    assert isinstance(server, FakeGymServer)
    assert server.n_actions == 3


def test_create_gym_server_requires_env_class() -> None:
    tr = MagicMock()
    tr.test.cmd_args.model_dump.return_value = {"live_rl_mode": True}

    with pytest.raises(ValueError, match="env_class"):
        _create_gym_server(tr)


def test_live_rl_mode_auto_detected(tmp_path: Path) -> None:
    """live_rl_mode=true in cmd_args builds the GymServer from env_class during __init__."""
    cmd_args = CmdArgs.model_validate(
        {"live_rl_mode": True, "env_class": f"{FakeGymServer.__module__}.FakeGymServer", "n_actions": 4}
    )
    tdef = TestDefinition(name="n", description="d", test_template_name="tt", cmd_args=cmd_args)
    test_run = TestRun(name="online_tr", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "out")

    env = CloudAIGymEnv(test_run=test_run, runner=MagicMock(spec=BaseRunner), rewards=RewardOverrides())

    assert env._is_online is True
    assert isinstance(env._gym_server, FakeGymServer)
    assert env._gym_server.n_actions == 4
