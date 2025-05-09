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

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml
from pydantic import ValidationError

from cloudai import (
    System,
    SystemConfigParsingError,
    Test,
    TestConfigParsingError,
    TestScenario,
    TestScenarioParsingError,
    format_validation_error,
)
from cloudai.registry import Registry
from cloudai.test_scenario_parser import TestScenarioParser

from .test_parser import TestParser

HOOK_ROOT = Path("conf/hook")
HOOK_TEST_ROOT = HOOK_ROOT / "test"


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

    @property
    def system(self) -> System:
        try:
            return self.parse_system(self.system_config_path)
        except SystemConfigParsingError:
            exit(1)  # exit right away to keep error message readable for users

    def parse(
        self,
        test_path: Optional[Path] = None,
        test_scenario_path: Optional[Path] = None,
    ) -> Tuple[System, List[Test], Optional[TestScenario]]:
        """
        Parse configurations for system, test templates, and test scenarios.

        Args:
            test_path (Optional[Path]): The file path for tests.
            test_scenario_path (Optional[Path]): The file path for the main test scenario.
                If None, all tests are included.

        Returns:
            Tuple[System, List[Test], Optional[TestScenario]]: A tuple containing the system object, a list of filtered
                test template objects, and the main test scenario object if provided.
        """
        tests: list[Test] = []
        if test_path:
            if not test_path.exists():
                raise FileNotFoundError(f"Test path '{test_path}' not found.")

            try:
                tests = self.parse_tests(list(test_path.glob("*.toml")), self.system)
            except TestConfigParsingError:
                exit(1)  # exit right away to keep error message readable for users

        if not HOOK_ROOT.exists():
            logging.debug(f"HOOK_ROOT path '{HOOK_ROOT}' does not exist.")

        try:
            hook_tests = (
                self.parse_tests(list(HOOK_TEST_ROOT.glob("*.toml")), self.system) if HOOK_TEST_ROOT.exists() else []
            )
        except TestConfigParsingError:
            exit(1)  # exit right away to keep error message readable for users

        if not test_scenario_path:
            all_tests = list({test.name: test for test in tests + hook_tests}.values())
            return self.system, all_tests, None

        test_mapping = {t.name: t for t in tests}
        hook_test_scenario_mapping = {}
        if HOOK_ROOT.exists() and list(HOOK_ROOT.glob("*.toml")):
            try:
                hook_test_scenario_mapping = self.parse_hooks(
                    list(HOOK_ROOT.glob("*.toml")), self.system, {t.name: t for t in hook_tests}
                )
            except TestScenarioParsingError:
                exit(1)  # exit right away to keep error message readable for users

        try:
            test_scenario = self.parse_test_scenario(
                test_scenario_path, self.system, test_mapping, hook_test_scenario_mapping
            )
        except TestScenarioParsingError:
            exit(1)  # exit right away to keep error message readable for users

        scenario_tests = {tr.test.name for tr in test_scenario.test_runs}
        hook_scenario_tests = {
            tr.test.name for hook_scenario in hook_test_scenario_mapping.values() for tr in hook_scenario.test_runs
        }

        relevant_test_names = scenario_tests.union(hook_scenario_tests)
        filtered_tests = [t for t in tests if t.name in relevant_test_names] + hook_tests
        filtered_tests = list({test.name: test for test in filtered_tests}.values())

        return self.system, filtered_tests, test_scenario

    @staticmethod
    def parse_hooks(hook_tomls: List[Path], system: System, test_mapping: Dict[str, Test]) -> Dict[str, TestScenario]:
        hook_mapping = {}
        for hook_test_scenario_path in hook_tomls:
            hook_scenario = Parser.parse_test_scenario(hook_test_scenario_path, system, test_mapping)
            hook_mapping[hook_scenario.name] = hook_scenario
        return hook_mapping

    @staticmethod
    def parse_test_scenario(
        test_scenario_path: Path,
        system: System,
        test_mapping: Dict[str, Test],
        hook_mapping: Optional[Dict[str, TestScenario]] = None,
        strict: bool = False,
    ) -> TestScenario:
        if hook_mapping is None:
            hook_mapping = {}

        test_scenario_parser = TestScenarioParser(test_scenario_path, system, test_mapping, hook_mapping, strict)
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
