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

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import toml
from pydantic import ValidationError

from cloudai import (
    Registry,
    System,
    SystemConfigParsingError,
    Test,
    TestConfigParsingError,
    TestParser,
    TestScenario,
    TestScenarioParser,
    TestScenarioParsingError,
    format_validation_error,
)


class Parser:
    """Main parser for parsing all types of configurations."""

    def __init__(self, system_config_path: Path) -> None:
        """
        Initialize a Parser instance.

        Args:
            system_config_path (str): The file path for system configurations.
        """
        logging.debug(f"Initializing parser with: {system_config_path=}")
        self.system_config_path = system_config_path

    def parse(
        self,
        test_path: Path,
        test_scenario_path: Optional[Path] = None,
    ) -> Tuple[System, List[Test], Optional[TestScenario]]:
        """
        Parse configurations for system, test templates, and test scenarios.

        Args:
            test_path (Path): The file path for tests.
            test_scenario_path (Optional[Path]): The file path for the main test scenario.
                If None, all tests are included.

        Returns:
            Tuple[System, List[Test], Optional[TestScenario]]: A tuple containing the system object, a list of filtered
                test template objects, and the main test scenario object if provided.
        """
        if not test_path.exists():
            raise FileNotFoundError(f"Test path '{test_path}' not found.")

        try:
            system = self.parse_system(self.system_config_path)
        except SystemConfigParsingError:
            exit(1)  # exit right away to keep error message readable for users

        try:
            tests = self.parse_tests(list(test_path.glob("*.toml")), system)
        except TestConfigParsingError:
            exit(1)  # exit right away to keep error message readable for users

        plugin_test_scenario_path = Path("conf/common/plugin")
        plugin_test_path = Path("conf/common/plugin/test")

        plugin_tests = (
            self.parse_tests(list(plugin_test_path.glob("*.toml")), system)
            if plugin_test_path and plugin_test_path.exists()
            else []
        )

        if test_scenario_path:
            return self._parse_with_scenario(system, tests, test_scenario_path, plugin_tests, plugin_test_scenario_path)

        return system, list(set(tests + plugin_tests)), None

    def _parse_with_scenario(
        self,
        system: System,
        tests: List[Test],
        test_scenario_path: Path,
        plugin_tests: List[Test],
        plugin_test_scenario_path: Optional[Path],
    ) -> Tuple[System, List[Test], Optional[TestScenario]]:
        """Parse tests and scenarios with a main test scenario path specified."""
        test_mapping = {t.name: t for t in tests}
        plugin_test_mapping = {t.name: t for t in plugin_tests}

        plugin_test_scenario_mapping = self._load_plugin_scenarios(plugin_test_scenario_path, plugin_test_mapping)
        test_scenario = self._load_main_scenario(test_scenario_path, test_mapping, plugin_test_scenario_mapping)

        all_used_test_names = self._collect_used_test_names(plugin_test_scenario_mapping, test_scenario)
        filtered_tests = [t for t in tests if t.name in all_used_test_names]

        return system, filtered_tests, test_scenario

    def _load_plugin_scenarios(
        self, plugin_test_scenario_path: Optional[Path], plugin_test_mapping: Dict[str, Test]
    ) -> Dict[str, TestScenario]:
        """Load plugin-specific test scenarios from the specified path."""
        if plugin_test_scenario_path and plugin_test_scenario_path.exists():
            try:
                return self.parse_plugins(list(plugin_test_scenario_path.glob("*.toml")), plugin_test_mapping)
            except TestScenarioParsingError:
                exit(1)
        return {}

    def _load_main_scenario(
        self,
        test_scenario_path: Path,
        test_mapping: Dict[str, Test],
        plugin_test_scenario_mapping: Dict[str, TestScenario],
    ) -> Optional[TestScenario]:
        """Load the main test scenario using provided mappings."""
        try:
            return self.parse_test_scenario(test_scenario_path, test_mapping, plugin_test_scenario_mapping)
        except TestScenarioParsingError:
            exit(1)

    def _collect_used_test_names(
        self, plugin_test_scenario_mapping: Dict[str, TestScenario], test_scenario: Optional[TestScenario]
    ) -> Set[str]:
        """Collect test names used in both plugin and main test scenarios."""
        # TODO: collect test names in the plugin test scenarios only
        plugin_test_names = {
            tr.test.name for scenario in plugin_test_scenario_mapping.values() for tr in scenario.test_runs
        }
        scenario_test_names = {tr.test.name for tr in test_scenario.test_runs} if test_scenario else set()
        return plugin_test_names.union(scenario_test_names)

    @staticmethod
    def parse_plugins(plugin_tomls: List[Path], test_mapping: Dict[str, Test]) -> Dict[str, TestScenario]:
        plugin_mapping = {}
        for plugin_test_scenario_path in plugin_tomls:
            plugin_scenario = Parser.parse_test_scenario(plugin_test_scenario_path, test_mapping)
            plugin_mapping[plugin_scenario.name] = plugin_scenario
        return plugin_mapping

    @staticmethod
    def parse_test_scenario(
        test_scenario_path: Path,
        test_mapping: Dict[str, Test],
        plugin_mapping: Optional[Dict[str, TestScenario]] = None,
    ) -> TestScenario:
        if plugin_mapping is None:
            plugin_mapping = {}

        test_scenario_parser = TestScenarioParser(test_scenario_path, test_mapping, plugin_mapping)
        test_scenario = test_scenario_parser.parse()
        return test_scenario

    @staticmethod
    def parse_tests(test_tomls: list[Path], system: System) -> list[Test]:
        test_parser = TestParser(test_tomls, system)
        tests: List[Test] = test_parser.parse_all()
        return tests

    @staticmethod
    def parse_system(system_config_path: Path) -> System:
        registry = Registry()
        with Path(system_config_path).open() as f:
            logging.debug(f"Opened system config file: {system_config_path}")
            data = toml.load(f)
            scheduler: Optional[str] = data.get("scheduler")
            if scheduler is None:
                logging.error(f"Missing 'scheduler' key in {system_config_path}")
                raise SystemConfigParsingError(f"Missing 'scheduler' key in {system_config_path}")

            if scheduler not in registry.systems_map:
                logging.error(
                    f"Unsupported system type '{scheduler}' in {system_config_path}. "
                    f"Should be one of: {', '.join(registry.systems_map.keys())}"
                )
                raise SystemConfigParsingError(
                    f"Unsupported system type '{scheduler}' in {system_config_path}. "
                    f"Supported types: {', '.join(registry.systems_map.keys())}"
                )

        try:
            system = registry.systems_map[scheduler](**data)
        except ValidationError as e:
            for err in e.errors(include_url=False):
                logging.error(f"Failed to parse system definition: {system_config_path}")
                err_msg = format_validation_error(err)
                logging.error(err_msg)
            raise SystemConfigParsingError("Failed to parse system definition") from e

        return system
