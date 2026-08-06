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
from typing import List, Optional, cast

import pytest
import toml
from pydantic import ValidationError

from cloudai.core import TestDefinition, TestRun, TestScenario
from cloudai.models.scenario import TestScenarioModel
from cloudai.test_scenario_parser import calculate_total_time_limit
from cloudai.workloads.nccl_test.nccl_comparison_report import ComparisonReportConfig


class DummyTestRun(TestRun):
    def __init__(self, time_limit: str) -> None:
        super().__init__(
            name="dummy_run",
            test=TestDefinition(
                name="dummy_test", description="dummy_test", test_template_name="dummy_test", cmd_args={}
            ),
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
        ([DummyHook([DummyTestRun("30m")])], None, None),
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


def test_report_spec_is_parsed() -> None:
    model = TestScenarioModel.model_validate(
        toml.loads("""
    name = "scenario"

    [reports]
    nccl_comparison = { enable = false, group_by = ["my_field"] }

    [[Tests]]
    id = "1"
    num_nodes = 2

    name = "name"
    description = "desc"
    test_template_name = "NcclTest"
    """)
    )

    assert len(model.reports) == 1
    cfg = cast(ComparisonReportConfig, model.reports["nccl_comparison"])
    assert cfg.enable is False
    assert cfg.group_by == ["my_field"]


def test_hf_local_home_path_is_parsed_as_a_test_definition_field() -> None:
    model = TestScenarioModel.model_validate(
        toml.loads("""
    name = "scenario"

    [[Tests]]
    id = "1"
    test_name = "vLLM"
    hf_local_home_path = "/raid/cloudai"
    """)
    )

    test_run = model.tests[0]
    assert test_run.hf_local_home_path == Path("/raid/cloudai")
    assert test_run.tdef_model_dump(by_alias=True)["hf_local_home_path"] == Path("/raid/cloudai")


def test_hf_local_home_path_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="hf_local_home_path must be an absolute path"):
        TestScenarioModel.model_validate(
            toml.loads("""
        name = "scenario"

        [[Tests]]
        id = "1"
        test_name = "vLLM"
        hf_local_home_path = "raid/cloudai"
        """)
        )
