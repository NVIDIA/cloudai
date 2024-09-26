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
from typing import List, Optional, Tuple

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
        self, test_path: Path, test_scenario_path: Optional[Path] = None
    ) -> Tuple[System, List[Test], Optional[TestScenario]]:
        """
        Parse configurations for system, test templates, and test scenarios.

        Returns
            Tuple[System, List[TestTemplate], TestScenario]: A tuple containing the system object, a list of test
                template objects, and the test scenario object.
        """
        if not test_path.exists():
            raise FileNotFoundError(f"Test path '{test_path}' not found.")

        try:
            system = self.parse_system(self.system_config_path)
        except SystemConfigParsingError:
            # exit right away to keep error message readable for users
            exit(1)

        test_parser = TestParser(test_path, system)
        try:
            tests: List[Test] = test_parser.parse_all()
        except TestConfigParsingError:
            # exit right away to keep error message readable for users
            exit(1)
        test_mapping = {t.name: t for t in tests}
        logging.debug(f"Parsed {len(tests)} tests: {[t.name for t in tests]}")

        filtered_tests = tests
        test_scenario: Optional[TestScenario] = None
        if test_scenario_path:
            test_scenario_parser = TestScenarioParser(str(test_scenario_path), system, test_mapping)
            try:
                test_scenario = test_scenario_parser.parse()
            except TestScenarioParsingError:
                # exit right away to keep error message readable for users
                exit(1)
            logging.debug("Parsed test scenario")

            scenario_tests = set(tr.test.name for tr in test_scenario.test_runs)
            filtered_tests = [t for t in tests if t.name in scenario_tests]

        return system, filtered_tests, test_scenario

    @staticmethod
    def parse_system(system_config_path: Path) -> System:
        registry = Registry()
        with Path(system_config_path).open() as f:
            logging.debug(f"Opened system config file: {system_config_path}")
            data = toml.load(f)
            scheduler = data.get("scheduler", "").lower()
            if scheduler not in registry.systems_map:
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
