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


from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from cloudai import Test, TestRun, TestScenarioParser
from cloudai._core.test_scenario_parser import _TestDependencyTOML, _TestRunTOML, _TestScenarioTOML


@pytest.fixture
def test_scenario_parser(tmp_path: Path) -> TestScenarioParser:
    tsp = TestScenarioParser("", {})
    return tsp


@pytest.fixture
def test() -> Test:
    return Test(
        name="t1",
        description="desc",
        test_template=Mock(),
        env_vars={},
        cmd_args={},
        extra_cmd_args="",
        extra_env_vars={},
    )


def test_single_test_case(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "template_test": "nccl"}]}
    )
    assert test_scenario.name == "nccl-test"
    assert len(test_scenario.test_runs) == 1
    assert test_scenario.job_status_check is True

    tr = test_scenario.test_runs[0]
    assert tr.name == "1"
    assert tr.iterations == 1
    assert tr.current_iteration == 0
    assert tr.dependencies == {}
    atest = test_scenario.test_runs[0].test
    assert atest.name == test.name
    assert atest.description == test.description
    assert atest.weight == 0
    assert atest.ideal_perf == 1.0
    assert atest.sol is None
    assert atest.test_template == test.test_template
    assert atest.env_vars == test.env_vars
    assert atest.cmd_args == test.cmd_args
    assert atest.extra_env_vars == test.extra_env_vars
    assert atest.extra_cmd_args == test.extra_cmd_args


@pytest.mark.parametrize(
    "prop,tvalue,cfg_value",
    [("sol", 1.0, 42.0), ("ideal_perf", 1.0, 42.0)],
)
def test_with_some_props(
    prop: str, tvalue: float, cfg_value: float, test: Test, test_scenario_parser: TestScenarioParser
) -> None:
    setattr(test, prop, tvalue)
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "template_test": "nccl", prop: cfg_value}]}
    )
    atest = test_scenario.test_runs[0].test
    val = getattr(atest, prop)
    assert val != tvalue
    assert val == cfg_value


def test_with_time_limit(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "template_test": "nccl", "time_limit": "10m"}]}
    )
    assert test_scenario.test_runs[0].time_limit == "10m"


def test_two_independent_cases(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test
    t2.name = "t2"

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "template_test": "nccl"}, {"id": "2", "template_test": "nccl2"}]}
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert test_scenario.test_runs[0].dependencies == {}

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].dependencies == {}


def test_raises_on_missing_mapping(test_scenario_parser: TestScenarioParser):
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_data({"name": "nccl-test", "Tests": [{"id": "1", "template_test": "nccl1"}]})
    assert exc_info.match("Test 'nccl1' not found in the test schema directory")


def test_raises_on_unknown_dependency(test_scenario_parser: TestScenarioParser) -> None:
    dep = _TestDependencyTOML(type="end_post_comp", id="dep")
    test_info = _TestRunTOML(id="test", template_test="nccl", dependencies=[dep])
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_dependencies_for_test(test_info)

    assert exc_info.match(f"Dependency section '{dep.id}' not found for " f"test '{test_info.id}'.")


def test_empty_dependency(test_scenario_parser: TestScenarioParser) -> None:
    test_info = _TestRunTOML(id="1", template_test="nccl")
    deps = test_scenario_parser._parse_dependencies_for_test(test_info)
    assert deps == {}


@pytest.mark.parametrize("time", [None, 1])
def test_returns_valid_dependency(time: Any, test_scenario_parser: TestScenarioParser, test: Test) -> None:
    dep = _TestDependencyTOML(type="end_post_comp", id="dep")
    test_info = _TestRunTOML(id="test", template_test="nccl", dependencies=[dep])
    if time is not None:
        dep.time = time
    test_scenario_parser.testruns_by_id = {"dep": TestRun("", test, 1, [])}
    deps = test_scenario_parser._parse_dependencies_for_test(test_info)

    assert len(deps) == 1
    assert "end_post_comp" in deps
    assert deps["end_post_comp"].test_run.test == test
    if time is not None:
        assert deps["end_post_comp"].time == time
    else:
        assert deps["end_post_comp"].time == 0


def test_cant_depends_on_itself(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    with pytest.raises(ValueError) as exc_info:
        _TestScenarioTOML.model_validate(
            {
                "name": "nccl-test",
                "Tests": [
                    {
                        "id": "1",
                        "template_test": "nccl",
                        "dependencies": [{"type": "end_post_comp", "id": "1"}],
                    },
                ],
            }
        )
    assert exc_info.match("Test '1' must not depend on itself.")


def test_two_dependent_cases(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test
    t2.name = "t2"

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {
            "name": "nccl-test",
            "Tests": [
                {"id": "1", "template_test": "nccl", "dependencies": [{"type": "end_post_comp", "id": "2"}]},
                {"id": "2", "template_test": "nccl2"},
            ],
        }
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert "end_post_comp" in test_scenario.test_runs[0].dependencies
    assert isinstance(test_scenario.test_runs[0].dependencies["end_post_comp"].test_run, TestRun)
    assert test_scenario.test_runs[0].dependencies["end_post_comp"].time == 0

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].dependencies == {}


def test_ids_must_be_unique(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_data(
            {
                "name": "nccl-test",
                "Tests": [
                    {"id": "1", "template_test": "nccl"},
                    {"id": "1", "template_test": "nccl"},
                ],
            }
        )
    assert exc_info.match("Duplicate test id '1' found in the test scenario.")


def test_list_of_tests_must_not_be_empty() -> None:
    with pytest.raises(ValueError) as exc_info:
        _TestScenarioTOML.model_validate({"name": "name"})
    assert exc_info.match("_TestScenarioTOML\nTests\n  Field required")

    with pytest.raises(ValueError) as exc_info:
        _TestScenarioTOML.model_validate({"name": "name", "Tests": []})
    assert exc_info.match("_TestScenarioTOML\nTests\n  List should have at least 1 item after validation")


def test_test_id_must_contain_at_least_one_letter() -> None:
    with pytest.raises(ValueError) as exc_info:
        _TestScenarioTOML.model_validate({"name": "name", "Tests": [{"id": "", "template_test": "nccl"}]})
    assert exc_info.match("_TestScenarioTOML\nTests.0.id\n  String should have at least 1 character")
