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
import logging
from typing import Any, Dict, Literal, Optional

import toml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from .system import System
from .test import Test, TestDependency
from .test_scenario import TestRun, TestScenario


class _TestDependencyTOML(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["end_post_comp"]
    id: str
    time: int = 0


class _TestRunTOML(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    template_test: str
    num_nodes: Optional[int] = None
    nodes: list[str] = Field(default_factory=list)
    weight: int = 0
    iterations: int = 1
    sol: Optional[float] = None
    ideal_perf: float = 1.0
    time_limit: Optional[str] = None
    dependencies: list[_TestDependencyTOML] = Field(default_factory=list)


class _TestScenarioTOML(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    job_status_check: bool = True
    tests: list[_TestRunTOML] = Field(default_factory=list, alias="Tests")

    @model_validator(mode="after")
    def check_circular_dependencies(self):
        """Check for circular dependencies in the test scenario."""
        for test_run in self.tests:
            for dep in test_run.dependencies:
                if dep.id == test_run.id:
                    raise ValueError(f"Test '{test_run.id}' must not depend on itself.")

        return self


class TestScenarioParser:
    """
    Parser for TestScenario objects.

    Attributes
        file_path (str): Path to the TOML configuration file.
        system: The system object to which the test scenarios apply.
        test_mapping: Mapping of test names to Test objects.
    """

    __test__ = False

    def __init__(self, file_path: str, system: System, test_mapping: Dict[str, Test]) -> None:
        self.file_path = file_path
        self.system = system
        self.test_mapping = test_mapping

    @staticmethod
    def format_validation_error(err: ErrorDetails) -> str:
        logging.error(f"Validation error: {err}")
        if err["msg"] == "Field required":
            return f"Field '{'.'.join(str(v) for v in err['loc'])}': {err['msg']}"

        return f"Field '{'.'.join(str(v) for v in err['loc'])}' with value '{err['input']}' is invalid: {err['msg']}"

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
        try:
            ts_model = _TestScenarioTOML.model_validate(data)
        except ValidationError as e:
            for err in e.errors(include_url=False):
                err_msg = self.format_validation_error(err)
                logging.error(err_msg)
            raise ValueError("Failed to parse Test Scenario definition") from e

        test_runs_by_id = {}
        for tr in ts_model.tests:
            if tr.id in test_runs_by_id:
                raise ValueError(f"Duplicate test id '{tr.id}' found in the test scenario.")
            test_runs_by_id[tr.id] = self._create_section_test_run(tr)

        total_weight = sum(tr.weight for tr in ts_model.tests)
        normalized_weight = 0 if total_weight == 0 else 100 / total_weight

        tests_data: dict[str, _TestRunTOML] = {}
        for tr in ts_model.tests:
            tests_data[tr.id] = tr

        for section, tr in test_runs_by_id.items():
            test_info = tests_data[section]

            deps = self._parse_dependencies_for_test(test_info, test_runs_by_id)
            tr.test.dependencies = deps

            tr.test.iterations = test_info.iterations
            tr.test.weight = test_info.weight * normalized_weight
            tr.test.sol = test_info.sol
            tr.test.ideal_perf = test_info.ideal_perf
            tr.time_limit = test_info.time_limit

        return TestScenario(
            name=ts_model.name, test_runs=list(test_runs_by_id.values()), job_status_check=ts_model.job_status_check
        )

    def _create_section_test_run(self, test_info: _TestRunTOML) -> TestRun:
        """
        Create a section-specific Test object by copying from the test mapping.

        Args:
            test_info (Dict[str, Any]): Information of the test.

        Returns:
            Test: Copied and updated Test object for the section.

        Raises:
            ValueError: If the test or nodes are not found within the system.
        """
        if test_info.template_test not in self.test_mapping:
            raise ValueError(
                f"Test '{test_info.template_test}' not found in the test schema directory. Please ensure that all "
                f"tests referenced in the test scenario schema exist in the test schema directory. To resolve this "
                f"issue, you can either add the corresponding test schema file for '{test_info.template_test}' in "
                f"the directory or remove the testreference from the test scenario schema."
            )

        original_test = self.test_mapping[test_info.template_test]

        test = Test(
            name=original_test.name,
            description=original_test.description,
            test_template=original_test.test_template,
            env_vars=copy.deepcopy(original_test.env_vars),
            cmd_args=copy.deepcopy(original_test.cmd_args),
            extra_env_vars=copy.deepcopy(original_test.extra_env_vars),
            extra_cmd_args=original_test.extra_cmd_args,
            dependencies=copy.deepcopy(original_test.dependencies),
            iterations=original_test.iterations,
            sol=original_test.sol,
            weight=original_test.weight,
            ideal_perf=original_test.ideal_perf,
        )

        tr = TestRun(test_info.id, test, num_nodes=test_info.num_nodes or 1, nodes=test_info.nodes)
        return tr

    def _parse_dependencies_for_test(
        self, test_info: _TestRunTOML, section_test_runs: Dict[str, TestRun]
    ) -> Dict[str, TestDependency]:
        """
        Parse and creates TestDependency objects for various types of dependencies, ignoring empty dependencies.

        Args:
            section (str): Section name of the test.
            test_info (_TestRunTOML): Information of the test.
            section_test_runs (Dict[str, TestRun]): Mapping of section names to TestRun objects.

        Returns:
            Dict[str, Optional[TestDependency]]: Parsed dependencies for the test.
        """
        dependencies = {}
        for dep in test_info.dependencies:
            dep_section = dep.id
            dep_test = section_test_runs.get(dep_section)
            if not dep_test:
                raise ValueError(f"Dependency section '{dep_section}' not found for " f"test '{test_info.id}'.")
            dep_time = dep.time
            dependencies[dep.type] = TestDependency(test=dep_test.test, time=dep_time)

        return dependencies
