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

import copy
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Type

import toml
from pydantic import ValidationError

from cloudai.util import deep_merge, format_time_limit, parse_time_limit

from .core import (
    MissingTestError,
    Registry,
    ReportGenerationStrategy,
    System,
    TestDependency,
    TestRun,
    TestScenario,
    TestScenarioParsingError,
    format_validation_error,
)
from .models.scenario import TestRunModel, TestScenarioModel
from .models.workload import TestDefinition
from .test_parser import TestParser


def get_reporters(test_info: TestRunModel, tdef: TestDefinition) -> Set[Type[ReportGenerationStrategy]]:
    reporters = copy.deepcopy(Registry().reports_map.get(type(tdef), set()))

    from .workloads.nccl_test import NCCLTestDefinition, NcclTestPredictionReportGenerationStrategy

    if isinstance(tdef, NCCLTestDefinition) and tdef.predictor is not None:
        reporters.add(NcclTestPredictionReportGenerationStrategy)

    return reporters


def calculate_total_time_limit(test_hooks: List[TestScenario], time_limit: Optional[str] = None) -> Optional[str]:
    if not time_limit:
        return None

    total_time = parse_time_limit(time_limit)
    total_time += sum(
        (
            parse_time_limit(test_run.time_limit)
            for hook in test_hooks
            for test_run in hook.test_runs
            if test_run.time_limit
        ),
        timedelta(),
    )

    return format_time_limit(total_time)


class TestScenarioParser:
    """
    Parser for TestScenario objects.

    Attributes
        file_path (Path): Path to the TOML configuration file.
        test_mapping: Mapping of test names to Test objects.
    """

    __test__ = False

    def __init__(
        self,
        file_path: Path,
        system: System,
        test_mapping: dict[str, TestDefinition],
        hook_mapping: dict[str, TestScenario],
    ) -> None:
        self.file_path = file_path
        self.system = system
        self.test_mapping: Mapping[str, TestDefinition] = test_mapping
        self.hook_mapping = hook_mapping

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
            ts_model = TestScenarioModel.model_validate(data)
        except ValidationError as e:
            logging.error(f"Failed to parse Test Scenario definition: {self.file_path}")
            for err in e.errors(include_url=False):
                err_msg = format_validation_error(err)
                logging.error(err_msg)
            raise TestScenarioParsingError("Failed to parse Test Scenario definition") from e

        total_weight = sum(tr.weight for tr in ts_model.tests)
        normalized_weight = 0 if total_weight == 0 else 100 / total_weight

        pre_test, post_test = None, None
        if ts_model.pre_test:
            pre_test = self.hook_mapping.get(ts_model.pre_test)
            if pre_test is None:
                msg = (
                    f"Pre-test hook '{ts_model.pre_test}' not found in hook mapping. "
                    "A corresponding hook should exist under 'conf/hook'. "
                    "Ensure that a proper hook directory is set under the working directory."
                )
                logging.error(msg)
                raise TestScenarioParsingError(msg)

        if ts_model.post_test:
            post_test = self.hook_mapping.get(ts_model.post_test)
            if post_test is None:
                msg = (
                    f"Post-test hook '{ts_model.post_test}' not found in hook mapping. "
                    "A corresponding hook should exist under 'conf/hook'. "
                    "Ensure that a proper hook directory is set under the working directory."
                )
                logging.error(msg)
                raise TestScenarioParsingError(msg)

        test_runs_by_id: dict[str, TestRun] = {
            trm.id: self._create_test_run(trm, normalized_weight, pre_test, post_test) for trm in ts_model.tests
        }

        tests_data: dict[str, TestRunModel] = {trm.id: trm for trm in ts_model.tests}
        for section, tr in test_runs_by_id.items():
            test_info = tests_data[section]
            tr.dependencies = {
                dep.type: TestDependency(test_run=test_runs_by_id[dep.id]) for dep in test_info.dependencies
            }

        return TestScenario(
            name=ts_model.name,
            test_runs=list(test_runs_by_id.values()),
            job_status_check=ts_model.job_status_check,
            reports=ts_model.reports,
        )

    def _create_test_run(
        self,
        test_info: TestRunModel,
        normalized_weight: float,
        pre_test: Optional[TestScenario] = None,
        post_test: Optional[TestScenario] = None,
    ) -> TestRun:
        """
        Create a section-specific Test object by copying from the test mapping.

        Args:
            test_info (Dict[str, Any]): Information of the test.
            normalized_weight (float): Normalized weight for the test.
            pre_test (Optional[TestScenario]): TestScenario object representing the pre-test sequence.
            post_test (Optional[TestScenario]): TestScenario object representing the post-test sequence.

        Returns:
            Test: Copied and updated Test object for the section.

        Raises:
            ValueError: If the test or nodes are not found within the system.
        """
        tdef = self._prepare_tdef(test_info)

        hooks = [hook for hook in [pre_test, post_test] if hook is not None]
        total_time_limit = calculate_total_time_limit(test_hooks=hooks, time_limit=test_info.time_limit)

        tr = TestRun(
            test_info.id,
            tdef,
            num_nodes=test_info.num_nodes or 1,
            iterations=test_info.iterations,
            nodes=test_info.nodes,
            time_limit=total_time_limit,
            sol=test_info.sol,
            weight=test_info.weight * normalized_weight,
            ideal_perf=test_info.ideal_perf,
            pre_test=pre_test,
            post_test=post_test,
            reports=get_reporters(test_info, tdef),
            extra_srun_args=test_info.extra_srun_args,
        )

        return tr

    def _prepare_tdef(self, test_info: TestRunModel) -> TestDefinition:
        tp = TestParser([self.file_path], self.system)
        tp.current_file = self.file_path

        if test_info.test_name:
            if test_info.test_name not in self.test_mapping:
                raise MissingTestError(test_info.test_name)
            test = self.test_mapping[test_info.test_name]

            test_defined = test.model_dump(by_alias=True)
            tc_defined = test_info.tdef_model_dump(by_alias=True)
            merged_data = deep_merge(test_defined, tc_defined)
            test = tp.load_test_definition(merged_data)
        elif test_info.test_template_name:  # test fully defined in the scenario
            test = tp._parse_data(test_info.tdef_model_dump(by_alias=True))
        else:
            # this should never happen, because we check for this in the modelvalidator
            raise ValueError(
                f"Cannot configure test case '{test_info.id}' with both 'test_name' and 'test_template_name'."
            )

        return test
