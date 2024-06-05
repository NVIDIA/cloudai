from pathlib import Path

import pytest
from cloudai import Parser
from cloudai._core.test_parser import TestParser


class Test_Parser:
    @pytest.fixture()
    def parser(self, tmp_path: Path) -> Parser:
        system = tmp_path / "system.toml"
        templates_dir = tmp_path / "templates"
        return Parser(system, templates_dir)

    def test_no_scenario_file(self, parser: Parser):
        scenario = parser.system_config_path.parent / "scenario.toml"
        tests_dir = parser.system_config_path.parent / "tests"
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse(tests_dir, scenario)
        assert "Test scenario path" in str(exc_info.value)

    def test_no_tests_dir(self, parser: Parser):
        scenario = parser.system_config_path.parent / "scenario.toml"
        scenario.touch()
        tests_dir = parser.system_config_path.parent / "tests"
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse(tests_dir, scenario)
        assert "Test path" in str(exc_info.value)


class TestTestParser:
    @pytest.fixture()
    def test_parser(self, tmp_path: Path) -> TestParser:
        return TestParser(tmp_path / "tests", {})

    def test_no_dir(self, test_parser: TestParser):
        with pytest.raises(FileNotFoundError) as exc_info:
            test_parser.load_test_names()
        assert "Test path" in str(exc_info.value)

    def test_not_a_dir(self, test_parser: TestParser):
        test_parser.directory_path.touch()
        with pytest.raises(NotADirectoryError) as exc_info:
            test_parser.load_test_names()
        assert "Test path" in str(exc_info.value)

    def test_one_config(self, test_parser: TestParser):
        test_parser.directory_path.mkdir()
        (test_parser.directory_path / "test1.toml").write_text("name = 'test1'")
        assert test_parser.load_test_names() == {"test1"}

    def test_many_configs(self, test_parser: TestParser):
        test_parser.directory_path.mkdir()
        (test_parser.directory_path / "test1.toml").write_text("name = 'test1'")
        (test_parser.directory_path / "test2.toml").write_text("name = 'test2'")
        assert test_parser.load_test_names() == {"test1", "test2"}
