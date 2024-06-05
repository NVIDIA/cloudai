# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Tuple

from .system import System
from .system_parser import SystemParser
from .test_parser import TestParser
from .test_scenario import TestScenario
from .test_scenario_parser import TestScenarioParser
from .test_template import TestTemplate
from .test_template_parser import TestTemplateParser

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        system_config_path: str,
        test_template_path: str,
        test_path: Optional[str] = None,
        test_scenario_path: Optional[str] = None,
    ) -> None:
        """
        Initialize a Parser instance.

        Args:
            system_config_path (str): The file path for system configurations.
            test_template_path (str): The file path for test_template configurations.
            test_path (str): The file path for test configurations.
            test_scenario_path (str): The file path for test scenario configurations.
        """
        self.system_config_path: str = system_config_path
        self.test_template_path: str = test_template_path
        self.test_path: Optional[str] = test_path
        self.test_scenario_path: Optional[str] = test_scenario_path
        logger.debug("Initialized with system and template paths")

    def parse_system_and_templates(self) -> Tuple[System, List[TestTemplate]]:
        """
        Parse system and test template configurations for installation purposes.

        Returns
            Tuple[System, List[TestTemplate]]: A tuple containing the system object
            and a list of test template objects.
        """
        system_parser = SystemParser(self.system_config_path)
        system = system_parser.parse()
        logger.debug("Parsed system config")

        test_template_parser = TestTemplateParser(system, self.test_template_path)
        test_templates = test_template_parser.parse_all()
        logger.debug(f"Parsed {len(test_templates)} test templates")

        return system, test_templates

    def parse(self) -> Tuple[System, List[TestTemplate], TestScenario]:
        """
        Parse configurations for system, test templates, and test scenarios.

        Returns
            Tuple[System, List[TestTemplate], TestScenario]: A tuple containing
            the system object, a list of test template objects, and the test scenario
            object.
        """
        system_parser = SystemParser(self.system_config_path)
        system = system_parser.parse()
        logger.debug("Parsed system config")

        test_template_parser = TestTemplateParser(system, self.test_template_path)
        test_templates = test_template_parser.parse_all()
        test_template_mapping = {t.name: t for t in test_templates}
        logger.debug(f"Parsed {len(test_templates)} test templates")

        assert self.test_path is not None, "Tests path must be provided for experiments."
        test_parser = TestParser(self.test_path, test_template_mapping)
        tests = test_parser.parse_all()
        test_mapping = {t.name: t for t in tests}
        logger.debug(f"Parsed {len(tests)} tests")

        assert self.test_scenario_path is not None, "Test scenarios path must be provided for experiments."
        test_scenario_parser = TestScenarioParser(self.test_scenario_path, system, test_mapping)
        test_scenario = test_scenario_parser.parse()
        logger.debug("Parsed test scenario")

        return system, test_templates, test_scenario
