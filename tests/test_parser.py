# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Generator, cast
from unittest.mock import Mock, patch

import pytest
from pydantic_core import ErrorDetails

from cloudai.core import Parser, Registry, Reporter, format_validation_error
from cloudai.models.scenario import ReportConfig, parse_reports_spec
from cloudai.systems.slurm.slurm_system import SlurmSystem


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

    @patch("cloudai.test_parser.TestParser.parse_all")
    def test_no_scenario(self, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"
        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        test_parser.return_value = fake_tests
        _, tests, _ = parser.parse(tests_dir, None)
        assert len(tests) == 3

    @patch("cloudai.test_parser.TestParser.parse_all")
    @patch("cloudai.test_scenario_parser.TestScenarioParser.parse")
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

    @patch("cloudai.test_parser.TestParser.parse_all")
    @patch("cloudai.test_scenario_parser.TestScenarioParser.parse")
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

    @patch("cloudai.test_parser.TestParser.parse_all")
    @patch("cloudai.test_scenario_parser.TestScenarioParser.parse")
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


class TestParseReportsSpec:
    @pytest.fixture(autouse=True, scope="class")
    def scenario_report(self) -> Generator[str, None, None]:
        class MyReporter(Reporter):
            def generate(self) -> None: ...

        rname = "scenario-test"
        Registry().add_scenario_report(rname, MyReporter, ReportConfig())

        yield rname

        Registry().scenario_reports.pop(rname)
        Registry().report_configs.pop(rname)

    def test_report_not_in_registry(self):
        with pytest.raises(ValueError) as exc_info:
            parse_reports_spec({"unknown": {}})
        assert "Report configuration for 'unknown' not found in the registry." in str(exc_info.value)
        assert "Available reports: " in str(exc_info.value)

    def test_scenario_reports_can_be_disallowed(self, scenario_report: str):
        with pytest.raises(ValueError) as exc_info:
            parse_reports_spec({scenario_report: {}}, allow_scenario_reports=False)
        assert f"Scenario level report '{scenario_report}' is not allowed here." in str(exc_info.value)

    def test_malformed_config_reported(self, scenario_report: str):
        with pytest.raises(ValueError) as exc_info:
            parse_reports_spec({scenario_report: {"enable": "invalid"}})
        assert f"Error validating report configuration '{scenario_report}' as ReportConfig: " in str(exc_info.value)
