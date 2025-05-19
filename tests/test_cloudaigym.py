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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cloudai._core.configurator.cloudai_gym import CloudAIGymEnv
from cloudai._core.configurator.grid_search import GridSearchAgent
from cloudai._core.runner import Runner
from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai._core.test_template_strategy import TestTemplateStrategy
from cloudai.systems import SlurmSystem
from cloudai.workloads.nemo_run import (
    Data,
    NeMoRunCmdArgs,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)
from cloudai.workloads.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy


@pytest.fixture
def nemorun() -> NeMoRunTestDefinition:
    return NeMoRunTestDefinition(
        name="NemoModel",
        description="Nemo Model",
        test_template_name="nemo_template",
        cmd_args=NeMoRunCmdArgs(docker_image_url="https://docker/url", task="some_task", recipe_name="some_recipe"),
    )


@pytest.fixture
def setup_env(slurm_system: SlurmSystem, nemorun: NeMoRunTestDefinition) -> tuple[TestRun, Runner]:
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000],
        val_check_interval=[100, 200],
        num_nodes=[1, 2],
        strategy=TrainerStrategy(
            tensor_model_parallel_size=[1, 2],
            pipeline_model_parallel_size=[1, 2],
            context_parallel_size=[2, 4],
        ),
    )
    tdef.cmd_args.data = Data(micro_batch_size=[1, 2])

    mock_command_gen = MagicMock()
    mock_command_gen.gen_srun_command.return_value = "srun mock command"
    mock_command_gen.generate_test_command.return_value = ["python", "run.py", "--arg", "value"]

    test_template_mock = MagicMock()
    test_template_mock.command_gen_strategy = mock_command_gen

    test_run = TestRun(
        name="mock_test_run",
        test=Test(tdef, test_template=test_template_mock),
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )

    test_scenario = TestScenario(name="mock_test_scenario", test_runs=[test_run])
    test_run.output_path = (
        slurm_system.output_path / test_scenario.name / test_run.name / f"{test_run.current_iteration}"
    )

    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    return test_run, runner


def test_observation_space(setup_env):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    observation_space = env.define_observation_space()

    expected_observation_space = [0.0]

    assert observation_space == expected_observation_space


def test_compute_reward():
    env = CloudAIGymEnv(test_run=MagicMock(), runner=MagicMock())

    observation = [0.34827126874999986]
    reward = env.compute_reward(observation)
    assert reward == pytest.approx(2.871, 0.001)

    observation = [0.0]
    reward = env.compute_reward(observation)
    assert reward == 0.0

    observation = []
    reward = env.compute_reward(observation)
    assert reward == 0.0


def test_tr_output_path(setup_env: tuple[TestRun, Runner]):
    test_run, runner = setup_env
    test_run.test.test_definition.cmd_args.data.global_batch_size = 8  # avoid constraint check failure
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    agent = GridSearchAgent(env)

    _, action = agent.select_action()
    env.test_run.step = 42
    env.step(action)

    assert env.test_run.output_path.name == "42"


def test_action_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2])
    )
    nemorun.cmd_args.data.micro_batch_size = [1, 2]
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2"]

    tr.test.test_definition = nemorun

    action_space = tr.param_space

    assert len(action_space) == 4
    assert action_space["data.micro_batch_size"] == nemorun.cmd_args.data.micro_batch_size
    assert action_space["trainer.max_steps"] == nemorun.cmd_args.trainer.max_steps
    assert (
        action_space["trainer.strategy.tensor_model_parallel_size"]
        == nemorun.cmd_args.trainer.strategy.tensor_model_parallel_size
    )
    assert action_space["extra_env_vars.DSE_VAR"] == nemorun.extra_env_vars["DSE_VAR"]


def test_all_combinations(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2]))
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2", "3"]
    tr.test.test_definition = nemorun

    real_combinations = tr.all_combinations
    assert len(real_combinations) == 6
    expected_combinations = [
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "3"},
    ]
    for expected in expected_combinations:
        assert expected in real_combinations, f"Expected {expected} in all_combinations"


def test_all_combinations_non_dse(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test.test_definition = nemorun
    assert len(tr.all_combinations) == 0


def test_all_combinations_non_dse_but_with_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test.test_definition = nemorun
    with patch.object(type(tr.test.test_definition), "is_dse_job", new_callable=PropertyMock(return_value=True)):
        assert len(tr.all_combinations) == 0


def test_params_set(setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    assert len(tr.all_combinations) > 1
    for action in tr.all_combinations:
        new_tr = tr.apply_params_set(action)
        cmd_args = TestTemplateStrategy._flatten_dict(new_tr.test.test_definition.cmd_args.model_dump())
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                assert new_tr.test.test_definition.extra_env_vars[key[len("extra_env_vars.") :]] == value
            else:
                assert cmd_args[key] == value


def test_params_set_validated(setup_env: tuple[TestRun, Runner], nemorun: NeMoRunTestDefinition):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000])
    tr.test.test_definition = nemorun
    action_space = tr.param_space
    action_space["trainer.max_steps"] = "invalid"

    with pytest.raises(UserWarning) as excinfo:
        tr.apply_params_set(action_space)

    assert excinfo.type is UserWarning
    assert "Pydantic serializer warnings:" in str(excinfo.value)
    assert "but got `str`" in str(excinfo.value)
