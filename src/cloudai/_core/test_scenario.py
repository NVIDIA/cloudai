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

from typing import List

from .test import Test


class TestScenario:
    """
    Represents a test scenario, comprising a set of tests.

    Attributes
        name (str): Unique name of the test scenario.
        tests (List[Test]): Tests in the scenario.
        job_status_check (bool): Flag indicating whether to check the job status or not.
    """

    __test__ = False

    def __init__(self, name: str, tests: List[Test], job_status_check: bool = True) -> None:
        """
        Initialize a TestScenario instance.

        Args:
            name (str): Name of the test scenario.
            tests (List[Test]): List of tests in the scenario.
            job_status_check (bool): Flag indicating whether to check the job status or not.
        """
        self.name = name
        self.tests = tests
        self.job_status_check = job_status_check

    def __repr__(self) -> str:
        """
        Return a string representation of the TestScenario instance.

        Returns
            str: String representation of the test scenario.
        """
        test_names = ", ".join([test.name for test in self.tests])
        return f"TestScenario(name={self.name}, tests=[{test_names}])"

    def pretty_print(self) -> str:
        """Print each test in the scenario along with its section name, description, and visualized dependencies."""
        s = f"Test Scenario: {self.name}\n"
        for test in self.tests:
            s += f"\nSection Name: {test.section_name}\n"
            s += f"  Test Name: {test.name}\n"
            s += f"  Description: {test.description}\n"
            if test.dependencies:
                for dep_type, dependency in test.dependencies.items():
                    if dependency:
                        s += (
                            f"  {dep_type.replace('_', ' ').title()}: {dependency.test.section_name}, "
                            f"Time: {dependency.time} seconds"
                        )
            else:
                s += "  No dependencies"
        return s
