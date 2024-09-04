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
from pydantic_core import ErrorDetails

from cloudai import (
    Registry,
    System,
    Test,
    TestParser,
    TestScenario,
    TestScenarioParser,
    TestTemplate,
    TestTemplateParser,
)


class Parser:
    """Main parser for parsing all types of configurations."""

    def __init__(self, system_config_path: Path, test_templates_dir: Path) -> None:
        """
        Initialize a Parser instance.

        Args:
            system_config_path (str): The file path for system configurations.
            test_templates_dir (str): The file path for test_template configurations.
        """
        logging.debug(f"Initializing parser with: {system_config_path=} {test_templates_dir=}")
        self.system_config_path = system_config_path
        self.test_template_path = test_templates_dir

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

        system = self.parse_system(self.system_config_path)

        test_template_parser = TestTemplateParser(system, self.test_template_path)
        test_templates: List[TestTemplate] = test_template_parser.parse_all()
        test_template_mapping = {t.name: t for t in test_templates}
        logging.debug(f"Parsed {len(test_templates)} test templates: {[t.name for t in test_templates]}")

        test_parser = TestParser(test_path, test_template_mapping)
        tests: List[Test] = test_parser.parse_all()
        test_mapping = {t.name: t for t in tests}
        logging.debug(f"Parsed {len(tests)} tests: {[t.name for t in tests]}")

        filtered_tests = tests
        test_scenario: Optional[TestScenario] = None
        if test_scenario_path:
            test_scenario_parser = TestScenarioParser(str(test_scenario_path), system, test_mapping)
            test_scenario = test_scenario_parser.parse()
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
                raise ValueError(
                    f"Unsupported system type '{scheduler}' in {system_config_path}. "
                    f"Supported types: {', '.join(registry.systems_map.keys())}"
                )

        try:
            system = registry.systems_map[scheduler](**data)
        except ValidationError as e:
            for err in e.errors(include_url=False):
                err_msg = Parser.format_validation_error(err)
                logging.error(err_msg)
            raise ValueError("Failed to parse system definition") from e

        return system

    @staticmethod
    def format_validation_error(err: ErrorDetails) -> str:
        logging.error(f"Validation error: {err}")
        if err["msg"] == "Field required":
            return f"Field '{'.'.join(str(v) for v in err['loc'])}': {err['msg']}"

        return f"Field '{'.'.join(str(v) for v in err['loc'])}' with value '{err['input']}' is invalid: {err['msg']}"
