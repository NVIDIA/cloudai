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

import pytest

from cloudai._core.configurator.agents.grid_search import GridSearchAgent
from cloudai._core.test_scenario import TestRun


@pytest.fixture
def mock_test_run():
    """
    Fixture to provide a mock TestRun object for testing.
    """
    test_run = MagicMock(spec=TestRun)
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
    return test_run


def test_grid_search_agent(mock_test_run):
    """
    Test the GridSearchAgent's ability to traverse the action space.
    """
    agent = GridSearchAgent(mock_test_run)
    agent.configure(config={})

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
