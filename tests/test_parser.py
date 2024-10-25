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
from typing import Dict, cast
from unittest.mock import Mock, patch

import pytest
from pydantic_core import ErrorDetails

from cloudai import Parser, TestScenario, format_validation_error
from cloudai.systems.slurm.slurm_system import SlurmSystem


class Test_Parser:
    @pytest.fixture()
    def parser(self, tmp_path: Path) -> Parser:
        system = Path.cwd() / "conf" / "common" / "system" / "standalone_system.toml"
        return Parser(system)

    def test_no_tests_dir(self, parser: Parser):
        tests_dir = parser.system_config_path.parent / "tests"
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse(tests_dir, None, None)
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
    def test_scenario_without_plugin(self, test_scenario_parser: Mock, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"

        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        test_parser.return_value = fake_tests

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        _, tests, _ = parser.parse(tests_dir, Path())

        assert len(tests) == 1
        assert tests[0].name == "test-1"

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    @patch("cloudai.parser.Parser.parse_plugins")
    def test_scenario_with_plugin_common_tests(
        self, parse_plugins: Mock, test_scenario_parser: Mock, test_parser: Mock, parser: Parser
    ):
        tests_dir = parser.system_config_path.parent.parent / "test"

        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        test_parser.return_value = fake_tests

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        fake_plugin = Mock()
        fake_plugin.test_runs = [Mock()]
        fake_plugin.test_runs[0].test.name = "test-1"
        parse_plugins.return_value = {"plugin-1": fake_plugin}

        _, tests, _ = parser.parse(tests_dir, Path(), Path())

        assert len(tests) == 1
        assert tests[0].name == "test-1"

    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    def test_scenario_with_plugin_exclusive_tests(self, test_scenario_parser: Mock, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"
        test_scenario_path = Path("/mock/test_scenario.toml")
        plugin_test_scenario_path = Path("/mock/plugin_scenarios")

        fake_tests = [Mock() for _ in range(4)]
        for i, test in enumerate(fake_tests):
            test.name = f"test-{i}"
        test_parser.return_value = fake_tests

        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"
        test_scenario_parser.return_value = fake_scenario

        fake_plugin_scenarios = {"plugin-1": Mock(test_runs=[Mock()])}
        fake_plugin_scenarios["plugin-1"].test_runs[0].test.name = "test-2"

        with patch.object(parser, "_load_plugin_scenarios", return_value=fake_plugin_scenarios):
            _, filtered_tests, _ = parser.parse(tests_dir, test_scenario_path, tests_dir, plugin_test_scenario_path)

        filtered_test_names = {t.name for t in filtered_tests}
        assert len(filtered_tests) == 2
        assert "test-1" in filtered_test_names
        assert "test-2" in filtered_test_names
        assert "test-0" not in filtered_test_names
        assert "test-3" not in filtered_test_names

    def test_collect_used_test_names(self, parser: Parser):
        fake_scenario = Mock()
        fake_scenario.test_runs = [Mock()]
        fake_scenario.test_runs[0].test.name = "test-1"

        fake_plugin_scenario_1 = Mock(spec=TestScenario)
        fake_plugin_scenario_1.test_runs = [Mock()]
        fake_plugin_scenario_1.test_runs[0].test.name = "test-2"

        fake_plugin_scenario_2 = Mock(spec=TestScenario)
        fake_plugin_scenario_2.test_runs = [Mock()]
        fake_plugin_scenario_2.test_runs[0].test.name = "test-3"

        fake_plugin_scenarios = cast(
            Dict[str, TestScenario], {"plugin-1": fake_plugin_scenario_1, "plugin-2": fake_plugin_scenario_2}
        )

        used_test_names = parser._collect_used_test_names(fake_plugin_scenarios, fake_scenario)
        assert used_test_names == {"test-1", "test-2", "test-3"}

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
