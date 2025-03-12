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

from unittest.mock import MagicMock, patch

import pytest

from cloudai._core.configurator.cloudai_gym import CloudAIGymEnv
from cloudai._core.runner import Runner
from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai.systems import SlurmSystem
from cloudai.workloads.nemo_run import (
    Data,
    NeMoRunCmdArgs,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)


@pytest.fixture
def setup_env(slurm_system: SlurmSystem):
    test_run = MagicMock(spec=TestRun)
    test_scenario = MagicMock(spec=TestScenario)

    cmd_args = NeMoRunCmdArgs(
        docker_image_url="https://docker/url",
        task="some_task",
        recipe_name="some_recipe",
        trainer=Trainer(
            max_steps=[1000, 2000],
            val_check_interval=[100, 200],
            num_nodes=[1, 2],
            strategy=TrainerStrategy(
                tensor_model_parallel_size=[1, 2],
                pipeline_model_parallel_size=[1, 2],
                context_parallel_size=[2, 4],
            ),
        ),
        data=Data(
            micro_batch_size=[1, 2],
        ),
    )

    test_definition = NeMoRunTestDefinition(
        name="NemoModel", description="Nemo Model", test_template_name="nemo_template", cmd_args=cmd_args
    )

    test_run.test = MagicMock()
    test_run.test.test_definition = test_definition

    test_run.name = "mock_test_run"
    test_scenario.name = "mock_test_scenario"
    test_scenario.test_runs = [test_run]

    runner = Runner(mode="run", system=slurm_system, test_scenario=test_scenario)

    return test_run, runner


def test_action_space_nemo(setup_env):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    action_space = env.define_action_space()

    expected_action_space = {
        "trainer.max_steps": 2,
        "trainer.val_check_interval": 2,
        "trainer.num_nodes": 2,
        "trainer.strategy.tensor_model_parallel_size": 2,
        "trainer.strategy.pipeline_model_parallel_size": 2,
        "trainer.strategy.context_parallel_size": 2,
        "trainer.strategy.virtual_pipeline_model_parallel_size": 1,
        "data.micro_batch_size": 2,
    }

    relevant_action_space = {key: action_space[key] for key in expected_action_space}
    assert set(relevant_action_space.keys()) == set(expected_action_space.keys())
    for key in expected_action_space:
        assert len(relevant_action_space[key]) == expected_action_space[key]


def test_observation_space(setup_env):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    observation_space = env.define_observation_space()

    expected_observation_space = [0.0]

    assert observation_space == expected_observation_space


def test_get_observation(tmp_path, setup_env):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner)

    output_path = tmp_path / "output" / "mock_test_scenario"
    output_path.mkdir(parents=True, exist_ok=True)
    subdir = output_path / "0"
    subdir.mkdir(parents=True, exist_ok=True)
    report_file_path = subdir / "0" / "report.txt"
    report_file_path.parent.mkdir(parents=True, exist_ok=True)
    report_file_path.write_text("Average: 0.34827126874999986\n")

    with patch.object(env, "parse_report", return_value=[0.34827126874999986]):
        observation = env.get_observation(action={})
        assert observation == [0.34827126874999986]


def test_parse_report(tmp_path):
    report_content = """Min: 0.342734
Max: 0.355174
Average: 0.34827126874999986
Median: 0.347785
Stdev: 0.0031025735345648264
"""
    report_file = tmp_path / "report.txt"
    report_file.write_text(report_content)

    env = CloudAIGymEnv(test_run=MagicMock(), runner=MagicMock())
    observation = env.parse_report(tmp_path)
    assert observation == [0.34827126874999986]


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


def test_populate_action_space():
    env = CloudAIGymEnv(test_run=MagicMock(), runner=MagicMock())
    action_space = {}
    cmd_args = NeMoRunCmdArgs(
        docker_image_url="https://docker/url",
        task="some_task",
        recipe_name="some_recipe",
        trainer=Trainer(
            num_nodes=[1, 2],
            strategy=TrainerStrategy(
                tensor_model_parallel_size=[1, 2],
                unknown_nested=[1, 2],  # type: ignore
            ),
        ),
        data=Data(
            micro_batch_size=[1, 2],
        ),
    )
    extra_env_args = {
        "extra_param_1": [10, 20]
    }
    combined_dict = {**cmd_args.model_dump(), **extra_env_args}
    env.populate_action_space("", combined_dict, action_space)

    assert action_space["trainer.num_nodes"] == [1, 2]
    assert action_space["trainer.strategy.tensor_model_parallel_size"] == [1, 2]
    assert action_space["trainer.strategy.unknown_nested"] == [1, 2]
    assert action_space["data.micro_batch_size"] == [1, 2]
    assert action_space["extra_param_1"] == [10, 20]

def test_populate_action_space_cmd_args_list():
    env = CloudAIGymEnv(test_run=MagicMock(), runner=MagicMock())
    action_space = {}
    cmd_args = NeMoRunCmdArgs(
        docker_image_url="https://docker/url",
        task="some_task",
        recipe_name="some_recipe",
        trainer=Trainer(
            num_nodes=[1, 2],
            strategy=TrainerStrategy(
                tensor_model_parallel_size=[1, 2],
            ),
        ),
        data=Data(
            micro_batch_size=[1, 2],
        ),
    )
    extra_env_args = {
        "extra_param_1": [10],
    }
    combined_dict = {**cmd_args.model_dump(), **extra_env_args}
    env.populate_action_space("", combined_dict, action_space)

    assert action_space["trainer.num_nodes"] == [1, 2]
    assert action_space["trainer.strategy.tensor_model_parallel_size"] == [1, 2]
    assert action_space["data.micro_batch_size"] == [1, 2]
    assert action_space["extra_param_1"] == [10]

def test_populate_action_space_extra_env_args_list():
    env = CloudAIGymEnv(test_run=MagicMock(), runner=MagicMock())
    action_space = {}
    cmd_args = NeMoRunCmdArgs(
        docker_image_url="https://docker/url",
        task="some_task",
        recipe_name="some_recipe",
        trainer=Trainer(
            num_nodes=1,
            strategy=TrainerStrategy(
                tensor_model_parallel_size=1,
            ),
        ),
        data=Data(
            micro_batch_size=1,
        ),
    )
    extra_env_args = {
        "extra_param_1": [10, 20],
    }
    combined_dict = {**cmd_args.model_dump(), **extra_env_args}
    env.populate_action_space("", combined_dict, action_space)

    assert action_space["extra_param_1"] == [10, 20]


