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

from typing import List

from .test import Test


class TestScenario:
    """
    Represents a test scenario, comprising a set of tests.

    Attributes:
        name (str): Unique name of the test scenario.
        tests (List[Test]): Tests in the scenario.
    """

    __test__ = False

    def __init__(self, name: str, tests: List[Test]) -> None:
        """
        Initializes a TestScenario instance.

        Args:
            name (str): Name of the test scenario.
            tests (List[Test]): List of tests in the scenario.
        """
        self.name = name
        self.tests = tests

    def __repr__(self) -> str:
        """
        Returns a string representation of the TestScenario instance.

        Returns:
            str: String representation of the test scenario.
        """
        test_names = ", ".join([test.name for test in self.tests])
        return f"TestScenario(name={self.name}, tests=[{test_names}])"

    def pretty_print(self) -> None:
        """
        Prints each test in the scenario along with its section name,
        description, and visualized dependencies.
        """
        print(f"Test Scenario: {self.name}")
        for test in self.tests:
            print(f"\nSection Name: {test.section_name}")
            print(f"  Test Name: {test.name}")
            print(f"  Description: {test.description}")
            if test.dependencies:
                for dep_type, dependency in test.dependencies.items():
                    if dependency:
                        print(
                            f"  {dep_type.replace('_', ' ').title()}: "
                            f"{dependency.test.section_name}, "
                            f"Time: {dependency.time} seconds"
                        )
            else:
                print("  No dependencies")
