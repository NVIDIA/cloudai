#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .system import System
from .system_parser import SystemParser
from .test import Test
from .test_parser import TestParser
from .test_scenario import TestScenario
from .test_scenario_parser import TestScenarioParser
from .test_template import TestTemplate
from .test_template_parser import TestTemplateParser


class Parser:
    """
    Main parser for parsing all types of configurations.

    Attributes
        system_config_path (str): The file path for system configurations.
        test_template_path (str): The file path for test template configurations.
        test_path (str): The file path for test configurations.
        test_scenario_path (str): The file path for test scenario configurations.
        logger (logging.Logger): Logger for the parser.
    """

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

        system_parser = SystemParser(str(self.system_config_path))
        system = system_parser.parse()
        logging.debug("Parsed system config")

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

            scenario_tests = set(t.name for t in test_scenario.tests)
            filtered_tests = [t for t in tests if t.name in scenario_tests]

        return system, filtered_tests, test_scenario
