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
from cloudai.systems.standalone_system import StandaloneSystem


@pytest.fixture
def test_scenario_parser(tmp_path: Path) -> TestScenarioParser:
    system = StandaloneSystem(name="test", install_path=tmp_path, output_path=tmp_path)
    tsp = TestScenarioParser("", system, {})
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
    test_scenario = test_scenario_parser._parse_data({"name": "nccl-test", "Tests": {"1": {"name": "nccl"}}})
    assert test_scenario.name == "nccl-test"
    assert len(test_scenario.test_runs) == 1
    assert test_scenario.job_status_check is True

    atest = test_scenario.test_runs[0].test
    assert atest.name == test.name
    assert atest.description == test.description
    assert atest.section_name == "Tests.1"
    assert atest.dependencies == {}
    assert atest.iterations == 1
    assert atest.current_iteration == 0
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
        {"name": "nccl-test", "Tests": {"1": {"name": "nccl", prop: cfg_value}}}
    )
    atest = test_scenario.test_runs[0].test
    val = getattr(atest, prop)
    assert val != tvalue
    assert val == cfg_value


def test_with_time_limit(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": {"1": {"name": "nccl", "time_limit": "10m"}}}
    )
    assert test_scenario.test_runs[0].time_limit == "10m"


def test_two_independent_cases(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test
    t2.name = "t2"

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": {"1": {"name": "nccl"}, "2": {"name": "nccl2"}}}
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert test_scenario.test_runs[0].test.dependencies == {}

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].test.dependencies == {}


def test_raises_on_missing_name(test_scenario_parser: TestScenarioParser):
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_data({"Tests": []})
    assert exc_info.match("Failed to parse Test Scenario definition")


def test_raises_on_missing_mapping(test_scenario_parser: TestScenarioParser):
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_data({"name": "nccl-test", "Tests": {"1": {"name": "nccl1"}}})
    assert exc_info.match("Test 'nccl1' not found in the test schema directory")


def test_raises_on_invalid_dependency_format(test_scenario_parser: TestScenarioParser) -> None:
    test_info = {"dependencies": {"dep": "should-be-dict-here"}}
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_dependencies_for_test("Tests.1", test_info, {})

    assert exc_info.match("Invalid format for dependency ")


def test_raises_on_unknown_dependency(test_scenario_parser: TestScenarioParser) -> None:
    test_info = {"dependencies": {"dep": {"unknown": "Tests.2"}}}
    with pytest.raises(ValueError) as exc_info:
        test_scenario_parser._parse_dependencies_for_test("Tests.1", test_info, {})

    assert exc_info.match("Dependency section .* not found")


def test_empty_dependency(test_scenario_parser: TestScenarioParser) -> None:
    deps = test_scenario_parser._parse_dependencies_for_test("Tests.1", {}, {})
    assert deps == {}

    test_info = {"dependencies": {}}
    deps = test_scenario_parser._parse_dependencies_for_test("Tests.1", test_info, {})
    assert deps == {}


@pytest.mark.parametrize("time", [None, "10m", 1, 1.5])
def test_returns_valid_dependency(time: Any, test_scenario_parser: TestScenarioParser, test: Test) -> None:
    test_info = {"dependencies": {"dep_type": {"name": "Tests.1"}}}
    if time is not None:
        test_info["dependencies"]["dep_type"]["time"] = time
    sec_runs = {"Tests.1": TestRun(test, 1, [])}
    deps = test_scenario_parser._parse_dependencies_for_test("Tests.1", test_info, sec_runs)

    assert len(deps) == 1
    assert "dep_type" in deps
    assert deps["dep_type"].test == test
    if time is not None:
        assert deps["dep_type"].time == time
    else:
        assert deps["dep_type"].time == 0


def test_two_dependent_tests(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test
    t2.name = "t2"

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {
            "name": "nccl-test",
            "Tests": {
                "1": {"name": "nccl", "dependencies": {"dep": {"name": "Tests.2"}}},
                "2": {"name": "nccl2"},
            },
        }
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert "dep" in test_scenario.test_runs[0].test.dependencies
    assert isinstance(test_scenario.test_runs[0].test.dependencies["dep"].test, Test)
    assert test_scenario.test_runs[0].test.dependencies["dep"].time == 0

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].test.dependencies == {}
