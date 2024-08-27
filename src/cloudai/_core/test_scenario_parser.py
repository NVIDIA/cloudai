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

import copy
import sys
from typing import Any, Dict

import toml

from .system import System
from .test import Test, TestDependency
from .test_scenario import TestScenario


class TestScenarioParser:
    """
    Parser for TestScenario objects.

    Attributes
        file_path (str): Path to the TOML configuration file.
        system: The system object to which the test scenarios apply.
        test_mapping: Mapping of test names to Test objects.
    """

    __test__ = False

    def __init__(
        self,
        file_path: str,
        system: System,
        test_mapping: Dict[str, Test],
    ) -> None:
        self.file_path = file_path
        self.system = system
        self.test_mapping = test_mapping

    def parse(self) -> TestScenario:
        """
        Parse the TOML file and returns a TestScenario object.

        Returns
            TestScenario: The parsed TestScenario object.
        """
        with open(self.file_path, "r") as file:
            data: Dict[str, Any] = toml.load(file)
            return self._parse_data(data)

    def _parse_data(self, data: Dict[str, Any]) -> TestScenario:
        """
        Parse data for a TestScenario object.

        Args:
            data (Dict[str, Any]): Data from a TOML file.

        Returns:
            TestScenario: Parsed TestScenario object.
        """
        if "name" not in data:
            raise KeyError("The 'name' field is missing from the data.")
        test_scenario_name = data["name"]
        job_status_check = data.get("job_status_check", True)
        raw_tests_data = data.get("Tests", {})
        tests_data = {f"Tests.{k}": v for k, v in raw_tests_data.items()}

        # Create section-specific test instances
        section_tests = {section: self._create_section_test(section, info) for section, info in tests_data.items()}

        total_weight = sum(test_info.get("weight", 0) for test_info in tests_data.values())
        normalized_weight = 0 if total_weight == 0 else 100 / total_weight

        # Update tests with dependencies
        for section, test in section_tests.items():
            test_info = tests_data[section]
            deps = self._parse_dependencies_for_test(section, test_info, section_tests)
            test.dependencies = deps

            # Parse and set iterations
            iterations = test_info.get("iterations", 1)
            test.iterations = iterations if isinstance(iterations, int) else sys.maxsize

            test.weight = test_info.get("weight", 0) * normalized_weight

            if "sol" in test_info:
                test.sol = test_info["sol"]

            if "ideal_perf" in test_info:
                test.ideal_perf = test_info["ideal_perf"]

            if "time_limit" in test_info:
                test.time_limit = test_info["time_limit"]

        return TestScenario(
            name=test_scenario_name, tests=list(section_tests.values()), job_status_check=job_status_check
        )

    def _create_section_test(self, section: str, test_info: Dict[str, Any]) -> Test:
        """
        Create a section-specific Test object by copying from the test mapping.

        Args:
            section (str): Section name of the test.
            test_info (Dict[str, Any]): Information of the test.

        Returns:
            Test: Copied and updated Test object for the section.

        Raises:
            ValueError: If the test or nodes are not found within the system.
        """
        test_name = test_info.get("name", "")
        if test_name not in self.test_mapping:
            raise ValueError(
                f"Test '{test_name}' not found in the test schema directory. Please ensure that all tests referenced "
                f"in the test scenario schema exist in the test schema directory. To resolve this issue, you can "
                f"either add the corresponding test schema file for '{test_name}' in the directory or remove the test "
                f"reference from the test scenario schema."
            )

        test = copy.deepcopy(self.test_mapping[test_name])
        test.test_template = self.test_mapping[test_name].test_template
        test.section_name = section
        test.num_nodes = int(test_info.get("num_nodes", 1))
        test.nodes = test_info.get("nodes", [])
        return test

    def _parse_dependencies_for_test(
        self,
        section: str,
        test_info: Dict[str, Any],
        section_tests: Dict[str, Test],
    ) -> Dict[str, TestDependency]:
        """
        Parse and creates TestDependency objects for various types of dependencies, ignoring empty dependencies.

        Args:
            section (str): Section name of the test.
            test_info (Dict[str, Any]): Information of the test.
            section_tests (Dict[str, Test]): Mapping of section names to Test objects.

        Returns:
            Dict[str, Optional[TestDependency]]: Parsed dependencies for the test.
        """
        dependencies = {}
        dep_info = test_info.get("dependencies", {})
        for dep_type, dep_details in dep_info.items():
            if dep_details:  # Check if dep_details is not empty
                if isinstance(dep_details, dict):
                    dep_section = dep_details.get("name", "")
                    dep_test = section_tests.get(dep_section)
                    if not dep_test:
                        raise ValueError(f"Dependency section '{dep_section}' not found for " f"test '{section}'.")
                    dep_time = dep_details.get("time", 0)
                    dependencies[dep_type] = TestDependency(test=dep_test, time=dep_time)
                else:
                    raise ValueError(f"Invalid format for dependency '{dep_type}' in " f"test '{section}'.")
            # Else, skip if dep_details is empty

        return dependencies
