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
from .test_scenario import TestRun, TestScenario


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
        section_test_runs = {
            section: self._create_section_test_run(section, info) for section, info in tests_data.items()
        }

        total_weight = sum(test_info.get("weight", 0) for test_info in tests_data.values())
        normalized_weight = 0 if total_weight == 0 else 100 / total_weight

        # Update tests with dependencies
        for section, tr in section_test_runs.items():
            test_info = tests_data[section]
            deps = self._parse_dependencies_for_test(section, test_info, section_test_runs)
            tr.test.dependencies = deps

            # Parse and set iterations
            iterations = test_info.get("iterations", 1)
            tr.test.iterations = iterations if isinstance(iterations, int) else sys.maxsize

            tr.test.weight = test_info.get("weight", 0) * normalized_weight

            if "sol" in test_info:
                tr.test.sol = test_info["sol"]

            if "ideal_perf" in test_info:
                tr.test.ideal_perf = test_info["ideal_perf"]

            if "time_limit" in test_info:
                tr.time_limit = test_info["time_limit"]

        return TestScenario(
            name=test_scenario_name, test_runs=list(section_test_runs.values()), job_status_check=job_status_check
        )

    def _create_section_test_run(self, section: str, test_info: Dict[str, Any]) -> TestRun:
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

        original_test = self.test_mapping[test_name]

        test = Test(
            name=original_test.name,
            description=original_test.description,
            test_template=original_test.test_template,
            cmd_args=copy.deepcopy(original_test.cmd_args),
            extra_env_vars=copy.deepcopy(original_test.extra_env_vars),
            extra_cmd_args=original_test.extra_cmd_args,
            dependencies=copy.deepcopy(original_test.dependencies),
            iterations=original_test.iterations,
            sol=original_test.sol,
            weight=original_test.weight,
            ideal_perf=original_test.ideal_perf,
        )

        test.section_name = section
        tr = TestRun(
            test,
            num_nodes=int(test_info.get("num_nodes", 1)),
            nodes=test_info.get("nodes", []),
        )
        return tr

    def _parse_dependencies_for_test(
        self,
        section: str,
        test_info: Dict[str, Any],
        section_test_runs: Dict[str, TestRun],
    ) -> Dict[str, TestDependency]:
        """
        Parse and creates TestDependency objects for various types of dependencies, ignoring empty dependencies.

        Args:
            section (str): Section name of the test.
            test_info (Dict[str, Any]): Information of the test.
            section_test_runs (Dict[str, TestRun]): Mapping of section names to TestRun objects.

        Returns:
            Dict[str, Optional[TestDependency]]: Parsed dependencies for the test.
        """
        dependencies = {}
        dep_info = test_info.get("dependencies", {})
        for dep_type, dep_details in dep_info.items():
            if dep_details:  # Check if dep_details is not empty
                if isinstance(dep_details, dict):
                    dep_section = dep_details.get("name", "")
                    dep_test = section_test_runs.get(dep_section)
                    if not dep_test:
                        raise ValueError(f"Dependency section '{dep_section}' not found for " f"test '{section}'.")
                    dep_time = dep_details.get("time", 0)
                    dependencies[dep_type] = TestDependency(test=dep_test.test, time=dep_time)
                else:
                    raise ValueError(f"Invalid format for dependency '{dep_type}' in " f"test '{section}'.")
            # Else, skip if dep_details is empty

        return dependencies
