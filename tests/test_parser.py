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
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from cloudai import Parser, format_validation_error
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.test_definitions.gpt import DSEValuesRange, GPTTestDefinition


class Test_Parser:
    @pytest.fixture()
    def parser(self, tmp_path: Path) -> Parser:
        system = Path.cwd() / "conf" / "common" / "system" / "standalone_system.toml"
        return Parser(system)

    def test_no_tests_dir(self, parser: Parser):
        tests_dir = parser.system_config_path.parent / "tests"
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse(tests_dir, None)
        assert "Test path" in str(exc_info.value)

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    def test_no_scenario(self, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"
        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        test_parser.return_value = fake_tests
        _, tests, _ = parser.parse(tests_dir, None)
        assert len(tests) == 3

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    def test_scenario_without_hook(self, test_scenario_parser: Mock, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"

        fake_tests = [Mock(name=f"test-{i}") for i in range(3)]
        for i, test in enumerate(fake_tests):
            test.name = f"test-{i}"

        test_parser.side_effect = [fake_tests, []]

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        _, tests, _ = parser.parse(tests_dir, Path())

        assert len(tests) == 1
        assert tests[0].name == "test-1"

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    @patch("cloudai.parser.Parser.parse_hooks")
    def test_scenario_with_hook_common_tests(
        self, parse_hooks: Mock, test_scenario_parser: Mock, test_parser: Mock, parser: Parser
    ):
        tests_dir = parser.system_config_path.parent.parent / "test"

        main_tests = [Mock() for _ in range(3)]
        for i, test in enumerate(main_tests):
            test.name = f"test-{i}"
        hook_tests = [Mock()]
        hook_tests[0].name = "test-1"

        test_parser.side_effect = [main_tests, hook_tests]

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        fake_hook = Mock()
        fake_hook.test_runs = [Mock()]
        fake_hook.test_runs[0].test.name = "test-1"
        parse_hooks.return_value = {"hook-1": fake_hook}

        _, tests, _ = parser.parse(tests_dir, Path())

        filtered_test_names = {"test-1"}
        assert len(tests) == 1
        assert "test-1" in filtered_test_names

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    def test_scenario_with_hook_exclusive_tests(self, test_scenario_parser: Mock, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"
        test_scenario_path = Path("/mock/test_scenario.toml")

        main_tests = [Mock() for _ in range(3)]
        hook_tests = [Mock()]
        for i, test in enumerate(main_tests):
            test.name = f"test-{i}"
        hook_tests[0].name = "hook-test-1"

        test_parser.side_effect = [main_tests, hook_tests]

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        _, filtered_tests, _ = parser.parse(tests_dir, test_scenario_path)

        filtered_test_names = {t.name for t in filtered_tests}
        assert len(filtered_tests) == 2
        assert "test-1" in filtered_test_names
        assert "hook-test-1" in filtered_test_names
        assert "test-0" not in filtered_test_names
        assert "test-2" not in filtered_test_names

    def test_parse_system(self, parser: Parser):
        parser.system_config_path = Path("conf/common/system/example_slurm_cluster.toml")
        system = cast(SlurmSystem, parser.parse_system(parser.system_config_path))

        assert len(system.partitions) == 2
        names = [partition.name for partition in system.partitions]
        assert "partition_1" in names
        assert "partition_2" in names

        assert len(system.groups) == 2
        assert "partition_1" in system.groups
        assert "partition_2" in system.groups

        # checking number of nodes in each partition
        assert len(system.partitions[0].slurm_nodes) == 100
        assert len(system.partitions[1].slurm_nodes) == 100

        # checking groups
        assert len(system.groups["partition_2"]) == 0
        assert len(system.groups["partition_1"]) == 4
        assert "group_1" in system.groups["partition_1"]
        assert "group_2" in system.groups["partition_1"]
        assert "group_3" in system.groups["partition_1"]
        assert "group_4" in system.groups["partition_1"]

        # checking number of nodes in each group
        assert len(system.groups["partition_1"]["group_1"]) == 25
        assert len(system.groups["partition_1"]["group_2"]) == 25
        assert len(system.groups["partition_1"]["group_3"]) == 25
        assert len(system.groups["partition_1"]["group_4"]) == 25

    @pytest.mark.parametrize(
        "error, expected_msg",
        [
            (
                ErrorDetails(type="missing", loc=("field",), msg="Field required", input=None),
                "Field 'field': Field required",
            ),
            (
                ErrorDetails(type="value_error", loc=("field", "subf"), msg="Invalid field", input="value"),
                "Field 'field.subf' with value 'value' is invalid: Invalid field",
            ),
        ],
    )
    def test_log_validation_errors_with_required_field_error(self, error: ErrorDetails, expected_msg: str):
        err_msg = format_validation_error(error)
        assert err_msg == expected_msg


GPT_TEST_DEFINITION = {
    "name": "n",
    "description": "d",
    "test_template_name": "ttn",
    "cmd_args": {
        "docker_image_url": "docker://url",
        "fdl_config": "/path",
    },
}


@pytest.mark.parametrize(
    "dse_field,value",
    [("fdl_config", ["/p1"]), ("fdl.num_groups", [1, 2])],
)
def test_dse_valid(dse_field: str, value: Any):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {dse_field: value}
    gpt = GPTTestDefinition(**data)

    assert isinstance(gpt, GPTTestDefinition)
    assert gpt.dse[dse_field] == value


@pytest.mark.parametrize("value", [1, "1", 1.0, 1j, None])
def test_dse_invalid_field_top_type(value: Any):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl_config": value}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 2
    assert "Input should be a valid dictionary or instance of DSEValuesRange" in errors
    assert "Input should be a valid list" in errors


@pytest.mark.parametrize("field", ["fdl_configA", "fdl.num_groupsA"])
def test_dse_raises_on_unknown_field(field: str):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {field: [1]}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 1
    assert errors[0] == f"Value error, 'GPTCmdArgs' doesn't have field '{field}'"


def test_dse_invalid_field_value_type():
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl_config": ["/p", 1]}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 1
    assert "Invalid type of value=1 ('int')" in errors[0]


@pytest.mark.parametrize(
    "input,output",
    [
        ({"start": 1, "end": 2}, DSEValuesRange(start=1, end=2)),
        ({"start": 1, "end": 2, "step": 0.5}, DSEValuesRange(start=1, end=2, step=0.5)),
    ],
)
def test_dse_range(input: dict, output: DSEValuesRange):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl.num_gpus": input}

    gpt = GPTTestDefinition(**data)

    assert gpt.dse["fdl.num_gpus"] == output
