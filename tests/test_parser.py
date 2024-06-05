from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cloudai import Parser


class Test_Parser:
    @pytest.fixture()
    def parser(self, tmp_path: Path) -> Parser:
        system = tmp_path / "system.toml"
        templates_dir = tmp_path / "templates"
        return Parser(system, templates_dir)

    def test_no_tests_dir(self, parser: Parser):
        tests_dir = parser.system_config_path.parent / "tests"
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse(tests_dir, None)
        assert "Test path" in str(exc_info.value)

    @patch("cloudai._core.system_parser.SystemParser.parse")
    @patch("cloudai._core.test_parser.TestParser.parse_all")
    def test_no_scenario(self, m_tp, m_sp, parser: Parser):
        tests_dir = parser.system_config_path.parent / "tests"
        tests_dir.mkdir()
        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        m_tp.return_value = fake_tests
        fake_scenario = Mock()
        fake_scenario.tests = [Mock()]
        fake_scenario.tests[0].name = "test-1"
        _, tests, _ = parser.parse(tests_dir, None)
        assert len(tests) == 3

    @patch("cloudai._core.system_parser.SystemParser.parse")
    @patch("cloudai._core.test_parser.TestParser.parse_all")
    @patch("cloudai._core.test_scenario_parser.TestScenarioParser.parse")
    def test_scenario_filters_tests(self, m_tsp, m_tp, m_sp, parser: Parser):
        tests_dir = parser.system_config_path.parent / "tests"
        tests_dir.mkdir()
        fake_tests = []
        for i in range(3):
            fake_tests.append(Mock())
            fake_tests[-1].name = f"test-{i}"
        m_tp.return_value = fake_tests
        fake_scenario = Mock()
        fake_scenario.tests = [Mock()]
        fake_scenario.tests[0].name = "test-1"
        m_tsp.return_value = fake_scenario
        _, tests, _ = parser.parse(tests_dir, Path())
        assert len(tests) == 1
