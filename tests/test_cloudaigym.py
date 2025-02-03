# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.fixture
def setup_env(slurm_system: SlurmSystem):
    test_run = MagicMock(spec=TestRun)
    test_scenario = MagicMock(spec=TestScenario)

    test_run.test = MagicMock()
    test_run.test.cmd_args = {
        "docker_image_url": "https://docker/url",
        "iters": [10, 100],
        "maxbytes": [1024, 2048],
        "minbytes": [512, 1024, 2048, 4096],
        "ngpus": [4],
        "subtest_name": "nccl_test",
        "warmup_iters": 5,
    }

    test_run.name = "mock_test_run"
    test_scenario.name = "mock_test_scenario"
    test_scenario.test_runs = [test_run]

    runner = Runner(mode="run", system=slurm_system, test_scenario=test_scenario)

    return test_run, runner


def test_action_space_nccl(setup_env):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner)
    action_space = env.define_action_space()

    expected_action_space = {
        "iters": 2,
        "maxbytes": 2,
        "minbytes": 4,
        "ngpus": 1,
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
