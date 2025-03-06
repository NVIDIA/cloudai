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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Type

if TYPE_CHECKING:
    from .report_generation_strategy import ReportGenerationStrategy
    from .test import Test


class TestDependency:
    """
    Represents a dependency for a test.

    Attributes
        test_run (TestRun): TestRun object it depends on.
    """

    __test__ = False

    def __init__(self, test_run: "TestRun") -> None:
        """
        Initialize a TestDependency instance.

        Args:
            test_run (TestRun): TestRun object it depends on.
        """
        self.test_run = test_run


@dataclass
class TestRun:
    __test__ = False

    name: str
    test: "Test"
    num_nodes: int
    nodes: List[str]
    output_path: Path = Path("")
    iterations: int = 1
    current_iteration: int = 0
    step: int = 0
    time_limit: Optional[str] = None
    sol: Optional[float] = None
    weight: float = 0.0
    ideal_perf: float = 1.0
    dependencies: dict[str, TestDependency] = field(default_factory=dict)
    pre_test: Optional["TestScenario"] = None
    post_test: Optional["TestScenario"] = None
    reports: Set[Type["ReportGenerationStrategy"]] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.name + self.test.name + str(self.iterations) + str(self.current_iteration))

    def has_more_iterations(self) -> bool:
        """
        Check if the test has more iterations to run.

        Returns
            bool: True if more iterations are pending, False otherwise.
        """
        return self.current_iteration < self.iterations


class TestScenario:
    """
    Represents a test scenario, comprising a set of tests.

    Attributes
        name (str): Unique name of the test scenario.
        tests (List[Test]): Tests in the scenario.
        job_status_check (bool): Flag indicating whether to check the job status or not.
    """

    __test__ = False

    def __init__(self, name: str, test_runs: List[TestRun], job_status_check: bool = True) -> None:
        """
        Initialize a TestScenario instance.

        Args:
            name (str): Name of the test scenario.
            test_runs (List[TestRun]): List of tests in the scenario with custom run options.
            job_status_check (bool): Flag indicating whether to check the job status or not.
        """
        self.name = name
        self.test_runs = test_runs
        self.job_status_check = job_status_check

    def __repr__(self) -> str:
        """
        Return a string representation of the TestScenario instance.

        Returns
            str: String representation of the test scenario.
        """
        test_names = ", ".join([tr.test.name for tr in self.test_runs])
        return f"TestScenario(name={self.name}, tests=[{test_names}])"

    def pretty_print(self) -> str:
        """Print each test in the scenario along with its section name, description, and visualized dependencies."""
        s = f"Test Scenario: {self.name}\n"
        for tr in self.test_runs:
            s += f"\nSection Name: {tr.name}\n"
            s += f"  Test Name: {tr.test.name}\n"
            s += f"  Description: {tr.test.description}\n"
            if tr.dependencies:
                for dep_type, dependency in tr.dependencies.items():
                    if dependency:
                        s += f"  {dep_type.replace('_', ' ').title()}: {dependency.test_run.name}"
            else:
                s += "  No dependencies"
        return s
