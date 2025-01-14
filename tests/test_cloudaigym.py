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

from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from cloudai._core.configurator.cloudai_gym import CloudAIGymEnv
from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai.systems import SlurmSystem


@pytest.fixture
def setup_env():
    test_run = MagicMock(spec=TestRun)
    system = MagicMock(spec=SlurmSystem)
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

    return test_run, system, test_scenario


def test_action_space_nccl(setup_env):
    test_run, system, test_scenario = setup_env
    env = CloudAIGymEnv(test_run=test_run, system=system, test_scenario=test_scenario)
    assert isinstance(env.action_space, Dict)

    expected_action_space = Dict(
        {
            "iters": Discrete(2),
            "maxbytes": Discrete(2),
            "minbytes": Discrete(4),
            "ngpus": Discrete(1),
        }
    )

    assert env.action_space.spaces.keys() == expected_action_space.spaces.keys()
    for key in expected_action_space.spaces:
        assert isinstance(env.action_space.spaces[key], Discrete)
        assert isinstance(expected_action_space.spaces[key], Discrete)
        assert env.action_space.spaces[key].__dict__ == expected_action_space.spaces[key].__dict__


def test_observation_space(setup_env):
    test_run, system, test_scenario = setup_env
    env = CloudAIGymEnv(test_run=test_run, system=system, test_scenario=test_scenario)
    assert isinstance(env.observation_space, Box)

    expected_observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    assert env.observation_space.shape == expected_observation_space.shape
    assert env.observation_space.dtype == expected_observation_space.dtype
    assert np.all(env.observation_space.low == expected_observation_space.low)
    assert np.all(env.observation_space.high == expected_observation_space.high)
