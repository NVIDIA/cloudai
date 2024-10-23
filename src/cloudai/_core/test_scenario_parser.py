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
from typing import Any, Dict, Literal, Optional

import toml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import TestScenarioParsingError, format_validation_error
from .test import Test
from .test_scenario import TestDependency, TestRun, TestScenario


class _TestDependencyTOML(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["end_post_comp", "start_post_init", "start_post_comp"]
    id: str


class _TestRunTOML(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    test_name: str
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
    tests: list[_TestRunTOML] = Field(alias="Tests", min_length=1)
    prologue: str = ""
    epilogue: str = ""

    @model_validator(mode="after")
    def check_no_self_dependency(self):
        """Check for circular dependencies in the test scenario."""
        for test_run in self.tests:
            for dep in test_run.dependencies:
                if dep.id == test_run.id:
                    raise ValueError(f"Test '{test_run.id}' must not depend on itself.")

        return self

    @model_validator(mode="after")
    def check_no_duplicate_ids(self):
        """Check for duplicate test ids in the test scenario."""
        test_ids = set()
        for tr in self.tests:
            if tr.id in test_ids:
                raise ValueError(f"Duplicate test id '{tr.id}' found in the test scenario.")
            test_ids.add(tr.id)

        return self

    @model_validator(mode="after")
    def check_all_dependencies_are_known(self):
        """Check that all dependencies are known."""
        test_ids = set(tr.id for tr in self.tests)
        for tr in self.tests:
            for dep in tr.dependencies:
                if dep.id not in test_ids:
                    raise ValueError(f"Dependency section '{dep.id}' not found for test '{tr.id}'.")

        return self


class TestScenarioParser:
    """
    Parser for TestScenario objects.

    Attributes
        file_path (Path): Path to the TOML configuration file.
        test_mapping: Mapping of test names to Test objects.
    """

    __test__ = False

    def __init__(self, file_path: Path, test_mapping: Dict[str, Test], plugin_mapping: Dict[str, TestScenario]) -> None:
        self.file_path = file_path
        self.test_mapping = test_mapping
        self.plugin_mapping = plugin_mapping

    def parse(self) -> TestScenario:
        """
        Parse the TOML file and return a TestScenario object.

        Returns
            TestScenario: The parsed TestScenario object.
        """
        with self.file_path.open("r") as file:
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
            logging.error(f"Failed to parse Test Scenario definition: {self.file_path}")
            for err in e.errors(include_url=False):
                err_msg = format_validation_error(err)
                logging.error(err_msg)
            raise TestScenarioParsingError("Failed to parse Test Scenario definition") from e

        total_weight = sum(tr.weight for tr in ts_model.tests)
        normalized_weight = 0 if total_weight == 0 else 100 / total_weight

        prologue_name = data.get("prologue", "")
        epilogue_name = data.get("epilogue", "")

        prologue = self.plugin_mapping.get(prologue_name, None) if prologue_name else None
        epilogue = self.plugin_mapping.get(epilogue_name, None) if epilogue_name else None

        testruns_by_id: dict[str, TestRun] = {
            tr.id: self._create_section_test_run(tr, normalized_weight, prologue, epilogue) for tr in ts_model.tests
        }

        tests_data: dict[str, _TestRunTOML] = {tr.id: tr for tr in ts_model.tests}
        for section, tr in testruns_by_id.items():
            test_info = tests_data[section]
            tr.dependencies = {
                dep.type: TestDependency(test_run=testruns_by_id[dep.id]) for dep in test_info.dependencies
            }

        return TestScenario(
            name=ts_model.name,
            test_runs=list(testruns_by_id.values()),
            job_status_check=ts_model.job_status_check,
        )

    def _create_section_test_run(
        self,
        test_info: _TestRunTOML,
        normalized_weight: float,
        prologue: Optional[TestScenario],
        epilogue: Optional[TestScenario],
    ) -> TestRun:
        """
        Create a section-specific Test object by copying from the test mapping.

        Args:
            test_info (Dict[str, Any]): Information of the test.
            normalized_weight (float): Normalized weight for the test.
            prologue (Optional[TestScenario]): TestScenario object representing the prologue sequence.
            epilogue (Optional[TestScenario]): TestScenario object representing the epilogue sequence.

        Returns:
            Test: Copied and updated Test object for the section.

        Raises:
            ValueError: If the test or nodes are not found within the system.
        """
        if test_info.test_name not in self.test_mapping:
            msg = (
                f"Test '{test_info.test_name}' not found in the test schema directory. Please ensure that all "
                f"tests referenced in the test scenario schema exist in the test schema directory. To resolve this "
                f"issue, you can either add the corresponding test schema file for '{test_info.test_name}' in "
                f"the directory or remove the test reference from the test scenario schema."
            )
            logging.error(f"Failed to parse Test Scenario definition: {self.file_path}")
            logging.error(msg)
            raise TestScenarioParsingError(msg)

        original_test = self.test_mapping[test_info.test_name]

        test = Test(test_definition=original_test.test_definition, test_template=original_test.test_template)

        tr = TestRun(
            test_info.id,
            test,
            num_nodes=test_info.num_nodes or 1,
            iterations=test_info.iterations,
            nodes=test_info.nodes,
            time_limit=test_info.time_limit,
            sol=test_info.sol,
            weight=test_info.weight * normalized_weight,
            ideal_perf=test_info.ideal_perf,
            prologue=prologue if prologue is not None else TestScenario(name="default_prologue", test_runs=[]),
            epilogue=epilogue if epilogue is not None else TestScenario(name="default_epilogue", test_runs=[]),
        )
        return tr
