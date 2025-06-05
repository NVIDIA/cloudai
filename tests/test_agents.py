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

from unittest.mock import MagicMock

import pytest

from cloudai.configurator import CloudAIGymEnv, GridSearchAgent


@pytest.fixture
def mock_env():
    """
    Fixture to provide a mock CloudAIGymEnv object for testing.
    """
    env = MagicMock(spec=CloudAIGymEnv)
    env.define_action_space.return_value = {
        "iters": [10, 100],
        "maxbytes": [1024, 2048],
        "minbytes": [512, 1024, 2048, 4096],
        "ngpus": [4],
    }
    return env


def test_grid_search_agent(mock_env):
    """
    Test the GridSearchAgent's ability to traverse the action space.
    """
    agent = GridSearchAgent(mock_env)
    agent.configure(config=mock_env.define_action_space.return_value)

    combinations = agent.get_all_combinations()

    expected_combinations = [
        {"iters": 10, "maxbytes": 1024, "minbytes": 512, "ngpus": 4},
        {"iters": 10, "maxbytes": 1024, "minbytes": 1024, "ngpus": 4},
        {"iters": 10, "maxbytes": 1024, "minbytes": 2048, "ngpus": 4},
        {"iters": 10, "maxbytes": 1024, "minbytes": 4096, "ngpus": 4},
        {"iters": 10, "maxbytes": 2048, "minbytes": 512, "ngpus": 4},
        {"iters": 10, "maxbytes": 2048, "minbytes": 1024, "ngpus": 4},
        {"iters": 10, "maxbytes": 2048, "minbytes": 2048, "ngpus": 4},
        {"iters": 10, "maxbytes": 2048, "minbytes": 4096, "ngpus": 4},
        {"iters": 100, "maxbytes": 1024, "minbytes": 512, "ngpus": 4},
        {"iters": 100, "maxbytes": 1024, "minbytes": 1024, "ngpus": 4},
        {"iters": 100, "maxbytes": 1024, "minbytes": 2048, "ngpus": 4},
        {"iters": 100, "maxbytes": 1024, "minbytes": 4096, "ngpus": 4},
        {"iters": 100, "maxbytes": 2048, "minbytes": 512, "ngpus": 4},
        {"iters": 100, "maxbytes": 2048, "minbytes": 1024, "ngpus": 4},
        {"iters": 100, "maxbytes": 2048, "minbytes": 2048, "ngpus": 4},
        {"iters": 100, "maxbytes": 2048, "minbytes": 4096, "ngpus": 4},
    ]

    assert combinations == expected_combinations
