# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.configurator import CloudAIGymEnv, GridSearchAgent
from cloudai.core import BaseRunner, Runner, TestRun, TestScenario
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
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
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
    env = CloudAIGymEnv(test_run=base_tr, runner=MagicMock())

    for input_value, expected_reward in test_cases:
        reward = env.compute_reward(input_value)
        assert reward == expected_reward


def test_compute_reward_invalid(base_tr: TestRun):
    base_tr.test.agent_reward_function = "nonexistent"

    with pytest.raises(KeyError) as exc_info:
        CloudAIGymEnv(test_run=base_tr, runner=MagicMock())

    assert "Reward function 'nonexistent' not found" in str(exc_info.value)
    assert (
        "Available functions: ['inverse', 'negative', 'identity', "
        "'ai_dynamo_weighted_normalized', 'ai_dynamo_ratio_normalized', 'ai_dynamo_log_scale']" in str(exc_info.value)
    )


def test_tr_output_path(setup_env: tuple[TestRun, BaseRunner]):
    test_run, runner = setup_env
    test_run.test.cmd_args.data.global_batch_size = 8  # avoid constraint check failure
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    agent = GridSearchAgent(env)

    _, action = agent.select_action()
    env.test_run.step = 42
    env.step(action)

    assert env.test_run.output_path.name == "42"


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
