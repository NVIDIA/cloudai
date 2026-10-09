# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from pathlib import Path
from typing import Callable, Generator, cast
from unittest.mock import Mock, patch

import pytest
import toml
from pydantic_core import ErrorDetails

from cloudai.core import (
    ConfigPaths,
    MissingTestError,
    Registry,
    Reporter,
    TestConfigParsingError,
    format_validation_error,
)
from cloudai.models.scenario import ReportConfig, parse_reports_spec
from cloudai.parser import Parser
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

    def test_custom_hook_root_is_used(self, parser: Parser, tmp_path: Path):
        hook_root = tmp_path / "hooks"
        parser = Parser(parser.system_config_path, hook_root)

        assert parser.hook_root == hook_root
        assert parser.hook_test_root == hook_root / "test"

    def test_custom_hook_root_is_used_by_parse(self, parser: Parser, tmp_path: Path):
        hook_root = tmp_path / "hooks"
        hook_test_root = hook_root / "test"
        hook_test_root.mkdir(parents=True)

        (hook_test_root / "custom_hook_test.toml").write_text(
            'name = "custom_hook_test"\n'
            'description = "custom hook test"\n'
            'test_template_name = "Sleep"\n'
            "\n"
            "[cmd_args]\n"
            "seconds = 1\n"
        )

        parser = Parser(parser.system_config_path, hook_root)

        _, tests, _ = parser.parse(None, None)

        assert "custom_hook_test" in {test.name for test in tests}

    @patch("cloudai.parser.Parser.parse_test_scenario")
    def test_parse_links_config_paths(self, parse_test_scenario: Mock, parser: Parser, tmp_path: Path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_scenario_path = tmp_path / "test_scenario.toml"
        parse_test_scenario.return_value = Mock(test_runs=[])
        parser = Parser(parser.system_config_path, tmp_path / "hooks")

        _, _, test_scenario = parser.parse(tests_dir, test_scenario_path)

        assert test_scenario is not None
        assert test_scenario.config_paths == ConfigPaths(
            system_path=parser.system_config_path.resolve(),
            tests_dir_path=tests_dir.resolve(),
            test_scenario_path=test_scenario_path.resolve(),
        )

    @patch("cloudai.parser.Parser.parse_test_scenario")
    def test_parse_links_config_paths_without_tests_dir(
        self, parse_test_scenario: Mock, parser: Parser, tmp_path: Path
    ):
        test_scenario_path = tmp_path / "test_scenario.toml"
        parse_test_scenario.return_value = Mock(test_runs=[])
        parser = Parser(parser.system_config_path, tmp_path / "hooks")

        _, _, test_scenario = parser.parse(None, test_scenario_path)

        assert test_scenario is not None
        assert test_scenario.config_paths is not None
        assert test_scenario.config_paths.tests_dir_path is None

    def test_custom_hook_root_is_used_for_hook_scenario_resolution(self, parser: Parser, tmp_path: Path):
        """A hook *scenario* toml (referenced via pre_test) must also be resolved from the custom hook_root,
        not just hook test tomls under `<hook_root>/test`."""
        hook_root = tmp_path / "hooks"
        hook_test_root = hook_root / "test"
        hook_test_root.mkdir(parents=True)

        (hook_test_root / "custom_hook_test.toml").write_text(
            'name = "custom_hook_test"\n'
            'description = "custom hook test"\n'
            'test_template_name = "Sleep"\n'
            "\n"
            "[cmd_args]\n"
            "seconds = 1\n"
        )
        (hook_root / "custom_hook_scenario.toml").write_text(
            'name = "custom_hook"\n\n[[Tests]]\nid = "Tests.hook"\ntest_name = "custom_hook_test"\n'
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "main_test.toml").write_text(
            'name = "main_test"\ndescription = "main test"\ntest_template_name = "Sleep"\n\n[cmd_args]\nseconds = 1\n'
        )
        test_scenario_path = tmp_path / "test_scenario.toml"
        test_scenario_path.write_text(
            'name = "main-scenario"\n'
            'pre_test = "custom_hook"\n\n'
            "[[Tests]]\n"
            'id = "Tests.main"\n'
            'test_name = "main_test"\n'
        )

        parser = Parser(parser.system_config_path, hook_root)

        _, _, test_scenario = parser.parse(tests_dir, test_scenario_path)

        assert test_scenario is not None
        pre_test = test_scenario.test_runs[0].pre_test
        assert pre_test is not None
        assert pre_test.name == "custom_hook"

    @patch("cloudai.test_parser.TestParser.parse_all")
    def test_no_scenario(self, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"
        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        test_parser.return_value = (fake_tests, {})
        _, tests, _ = parser.parse(tests_dir, None)
        assert len(tests) == 3

    @patch("cloudai.test_parser.TestParser.parse_all")
    @patch("cloudai.test_scenario_parser.TestScenarioParser.parse")
    def test_scenario_without_hook(self, test_scenario_parser: Mock, test_parser: Mock, parser: Parser):
        tests_dir = parser.system_config_path.parent.parent / "test"

        fake_tests = [Mock(name=f"test-{i}") for i in range(3)]
        for i, test in enumerate(fake_tests):
            test.name = f"test-{i}"

        test_parser.side_effect = [(fake_tests, {}), ([], {})]

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

        test_parser.side_effect = [(main_tests, {}), (hook_tests, {})]

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

        test_parser.side_effect = [(main_tests, {}), (hook_tests, {})]

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

    @pytest.mark.parametrize("install_path", ["absolute", "install", "../install_tmp"])
    def test_parse_system(self, parser: Parser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_path: str):
        system_data = toml.load("conf/common/system/example_slurm_cluster.toml")
        system_data["install_path"] = str(tmp_path / "install") if install_path == "absolute" else install_path
        parser.system_config_path = tmp_path / "system.toml"
        parser.system_config_path.write_text(toml.dumps(system_data))
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        system = cast(SlurmSystem, parser.parse_system(parser.system_config_path))

        assert system.install_path == Path(system_data["install_path"]).absolute()
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


class TestUnparseableTestConfigs:
    @pytest.fixture()
    def tests_dir(
        self,
        tmp_path: Path,
        write_parseable_test: Callable[[Path, str], None],
        write_unregistered_agent_test: Callable[[Path, str], None],
    ) -> Path:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        write_parseable_test(tests_dir / "good_test.toml", "good_test")
        write_unregistered_agent_test(tests_dir / "bad_agent_test.toml", "bad_agent_test")
        (tests_dir / "broken_toml_test.toml").write_text('name = "broken_toml_test"\ndescription = \n')
        return tests_dir

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> Parser:
        system = Path.cwd() / "conf" / "common" / "system" / "standalone_system.toml"
        return Parser(system, tmp_path / "hooks")

    @pytest.fixture()
    def raising_parser(self, tmp_path: Path) -> Parser:
        system = Path.cwd() / "conf" / "common" / "system" / "standalone_system.toml"
        return Parser(system, tmp_path / "hooks", exit_on_error=False)

    def test_parse_tests_still_raises_on_any_unparseable_config(self, parser: Parser, tests_dir: Path):
        with pytest.raises(TestConfigParsingError):
            Parser.parse_tests(sorted(tests_dir.glob("*.toml")), parser.system)

    def test_parse_tolerates_unparseable_config_the_scenario_does_not_reference(
        self,
        parser: Parser,
        tests_dir: Path,
        write_scenario: Callable[[Path, str], None],
        caplog: pytest.LogCaptureFixture,
    ):
        test_scenario_path = tests_dir.parent / "test_scenario.toml"
        write_scenario(test_scenario_path, "good_test")

        with caplog.at_level(logging.DEBUG):
            _, tests, test_scenario = parser.parse(tests_dir, test_scenario_path)

        assert test_scenario is not None
        assert [test.name for test in tests] == ["good_test"]
        warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert [msg for msg in warnings if "could not be parsed" in msg] == [
            f"2 test config(s) under '{tests_dir}' could not be parsed and are not available to this scenario. "
            "Run verify-configs for the details."
        ]
        debug = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
        assert any(str(tests_dir / "bad_agent_test.toml") in msg and "is not registered" in msg for msg in debug)
        assert any(str(tests_dir / "broken_toml_test.toml") in msg and "TOML parsing error" in msg for msg in debug)

    def test_parse_raises_test_config_parsing_error_naming_the_referenced_unparseable_config(
        self, raising_parser: Parser, tests_dir: Path, write_scenario: Callable[[Path, str], None]
    ):
        test_scenario_path = tests_dir.parent / "test_scenario.toml"
        write_scenario(test_scenario_path, "bad_agent_test")

        with pytest.raises(TestConfigParsingError) as exc_info:
            raising_parser.parse(tests_dir, test_scenario_path)

        msg = str(exc_info.value)
        assert str(tests_dir / "bad_agent_test.toml") in msg
        assert "is not registered" in msg

    def test_parse_raises_missing_test_when_the_scenario_names_a_test_that_exists_nowhere(
        self, raising_parser: Parser, tests_dir: Path, write_scenario: Callable[[Path, str], None]
    ):
        test_scenario_path = tests_dir.parent / "test_scenario.toml"
        write_scenario(test_scenario_path, "no_such_test_anywhere")

        with pytest.raises(MissingTestError) as exc_info:
            raising_parser.parse(tests_dir, test_scenario_path)

        assert exc_info.value.test_name == "no_such_test_anywhere"

    def test_parse_without_scenario_exits_on_any_unparseable_config(self, parser: Parser, tests_dir: Path):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse(tests_dir, None)

        assert exc_info.value.code == 1

    def test_parse_exits_when_a_hook_test_is_unparseable(
        self,
        parser: Parser,
        tmp_path: Path,
        tests_dir: Path,
        write_unregistered_agent_test: Callable[[Path, str], None],
        write_scenario: Callable[[Path, str], None],
    ):
        (tests_dir / "bad_agent_test.toml").unlink()
        (tests_dir / "broken_toml_test.toml").unlink()
        hook_test_root = parser.hook_test_root
        hook_test_root.mkdir(parents=True)
        write_unregistered_agent_test(hook_test_root / "bad_hook_test.toml", "bad_hook_test")
        test_scenario_path = tmp_path / "test_scenario.toml"
        write_scenario(test_scenario_path, "good_test")

        with pytest.raises(SystemExit) as exc_info:
            parser.parse(tests_dir, test_scenario_path)

        assert exc_info.value.code == 1
