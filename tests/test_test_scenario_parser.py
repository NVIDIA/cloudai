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
from typing import List, Optional
from unittest.mock import create_autospec

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai._core.test_scenario_parser import calculate_total_time_limit
from tests.conftest import MyTestDefinition


class DummyTestRun(TestRun):
    def __init__(self, time_limit: str) -> None:
        dummy_test = create_autospec(Test, instance=True)
        dummy_test.name = "dummy_test"
        dummy_test.test_definition = MyTestDefinition
        super().__init__(
            name="dummy_run",
            test=dummy_test,
            num_nodes=1,
            nodes=[],
            output_path=Path(""),
            iterations=1,
            current_iteration=0,
            step=0,
            time_limit=time_limit,
            sol=None,
            weight=0.0,
            ideal_perf=1.0,
            dependencies={},
            pre_test=None,
            post_test=None,
            reports=set(),
        )


class DummyHook(TestScenario):
    def __init__(self, test_runs: List[TestRun]) -> None:
        super().__init__(name="dummy", test_runs=test_runs)


@pytest.mark.parametrize(
    "test_hooks, time_limit, expected",
    [
        ([], None, None),
        ([], "1h", "01:00:00"),
        ([DummyHook([DummyTestRun("30m")])], "1h", "01:30:00"),
        ([DummyHook([DummyTestRun("15m")]), DummyHook([DummyTestRun("45m")])], "1h", "02:00:00"),
        ([DummyHook([DummyTestRun("1h")])], "1-00:00:00", "1-01:00:00"),
    ],
)
def test_calculate_total_time_limit(
    test_hooks: List[TestScenario], time_limit: Optional[str], expected: Optional[str]
) -> None:
    assert calculate_total_time_limit(test_hooks, time_limit) == expected
