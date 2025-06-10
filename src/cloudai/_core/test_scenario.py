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

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Type, Union

from .system import System
from .test_template_strategy import TestTemplateStrategy

if TYPE_CHECKING:
    from ..models.scenario import ReportConfig
    from .report_generation_strategy import ReportGenerationStrategy
    from .test import Test


METRIC_ERROR = -1.0


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
    num_nodes: Union[int, list[int]]
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
    pre_test: Optional[TestScenario] = None
    post_test: Optional[TestScenario] = None
    reports: Set[Type[ReportGenerationStrategy]] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.name + self.test.name + str(self.iterations) + str(self.current_iteration))

    def has_more_iterations(self) -> bool:
        """
        Check if the test has more iterations to run.

        Returns
            bool: True if more iterations are pending, False otherwise.
        """
        return self.current_iteration + 1 < self.iterations

    @property
    def metric_reporter(self) -> Optional[Type[ReportGenerationStrategy]]:
        if not self.reports:
            return None

        for r in self.reports:
            if self.test.test_definition.agent_metric in r.metrics:
                return r

        return None

    def get_metric_value(self, system: System) -> float:
        report = self.metric_reporter
        if report is None:
            return METRIC_ERROR

        return report(system, self).get_metric(self.test.test_definition.agent_metric)

    @property
    def is_dse_job(self) -> bool:
        return self.test.test_definition.is_dse_job or isinstance(self.num_nodes, list)

    @property
    def nnodes(self) -> int:
        """Type safe getter for num_nodes, should only be used on an unrolled DSE job."""
        if isinstance(self.num_nodes, list):
            raise TypeError("num_nodes is a list, cannot be used as a scalar.")
        return self.num_nodes

    @property
    def param_space(self) -> dict[str, Any]:
        cmd_args_dict = TestTemplateStrategy._flatten_dict(self.test.test_definition.cmd_args.model_dump())
        extra_env_vars_dict = self.test.test_definition.extra_env_vars

        action_space: dict[str, Any] = {
            **{key: value for key, value in cmd_args_dict.items() if isinstance(value, list)},
            **{f"extra_env_vars.{key}": value for key, value in extra_env_vars_dict.items() if isinstance(value, list)},
        }
        if isinstance(self.num_nodes, list):
            action_space["NUM_NODES"] = self.num_nodes

        return action_space

    @property
    def all_combinations(self) -> list[dict[str, Any]]:
        if not self.is_dse_job:
            return []

        param_space: dict[str, Any] = self.param_space
        if not param_space:
            return []

        parameter_values: list[Any] = []
        for _, values in param_space.items():
            parameter_values.append(values)
        action_combinations = list(itertools.product(*parameter_values))

        keys = list(param_space.keys())
        all_combinations = [dict(zip(keys, combination, strict=True)) for combination in action_combinations]

        return all_combinations

    def apply_params_set(self, action: dict[str, Any]) -> "TestRun":
        tdef = self.test.test_definition.model_copy(deep=True)
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                tdef.extra_env_vars[key[len("extra_env_vars.") :]] = value
            else:
                attrs = key.split(".")
                obj = tdef.cmd_args
                for attr in attrs[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attrs[-1], value)

        new_tr = copy.deepcopy(self)
        if "NUM_NODES" in action:
            new_tr.num_nodes = action["NUM_NODES"]
        new_tr.test.test_definition = type(tdef)(**tdef.model_dump())  # re-create the model to enable validation
        return new_tr


class TestScenario:
    """
    Represents a test scenario, comprising a set of tests.

    Attributes
        name (str): Unique name of the test scenario.
        tests (List[Test]): Tests in the scenario.
        job_status_check (bool): Flag indicating whether to check the job status or not.
    """

    __test__ = False

    def __init__(
        self,
        name: str,
        test_runs: List[TestRun],
        job_status_check: bool = True,
        report_configs: Optional[dict[str, ReportConfig]] = None,
    ) -> None:
        """
        Initialize a TestScenario instance.

        Args:
            name (str): Name of the test scenario.
            test_runs (List[TestRun]): List of tests in the scenario with custom run options.
            job_status_check (bool): Flag indicating whether to check the job status or not.
            report_configs (Optional[dict[str, ReportConfig]]): Optional dictionary of report configurations.
        """
        self.name = name
        self.test_runs = test_runs
        self.job_status_check = job_status_check
        self.report_configs = report_configs

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
