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

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import toml
from pydantic import ValidationError

from ..models.scenario import TestRunModel, TestScenarioModel
from ..models.workload import TestDefinition
from ..workloads.chakra_replay import ChakraReplayReportGenerationStrategy, ChakraReplayTestDefinition
from ..workloads.jax_toolbox import (
    GPTTestDefinition,
    GrokTestDefinition,
    JaxToolboxReportGenerationStrategy,
    NemotronTestDefinition,
)
from ..workloads.megatron_run import CheckpointTimingReportGenerationStrategy, MegatronRunTestDefinition
from ..workloads.nccl_test import NCCLTestDefinition, NcclTestReportGenerationStrategy
from ..workloads.nemo_launcher import NeMoLauncherReportGenerationStrategy, NeMoLauncherTestDefinition
from ..workloads.nemo_run import NeMoRunReportGenerationStrategy, NeMoRunTestDefinition
from ..workloads.sleep import SleepReportGenerationStrategy, SleepTestDefinition
from ..workloads.slurm_container import SlurmContainerReportGenerationStrategy, SlurmContainerTestDefinition
from ..workloads.ucc_test import UCCTestDefinition, UCCTestReportGenerationStrategy
from .exceptions import TestScenarioParsingError, format_validation_error
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test import Test
from .test_parser import TestParser
from .test_scenario import TestDependency, TestRun, TestScenario

DEFAULT_REPORTERS: dict[Type[TestDefinition], Set[Type[ReportGenerationStrategy]]] = {
    ChakraReplayTestDefinition: {ChakraReplayReportGenerationStrategy},
    GPTTestDefinition: {JaxToolboxReportGenerationStrategy},
    GrokTestDefinition: {JaxToolboxReportGenerationStrategy},
    MegatronRunTestDefinition: {CheckpointTimingReportGenerationStrategy},
    NCCLTestDefinition: {NcclTestReportGenerationStrategy},
    NeMoLauncherTestDefinition: {NeMoLauncherReportGenerationStrategy},
    NeMoRunTestDefinition: {NeMoRunReportGenerationStrategy},
    NemotronTestDefinition: {JaxToolboxReportGenerationStrategy},
    SleepTestDefinition: {SleepReportGenerationStrategy},
    SlurmContainerTestDefinition: {SlurmContainerReportGenerationStrategy},
    UCCTestDefinition: {UCCTestReportGenerationStrategy},
}


def get_reporters(test_info: TestRunModel, tdef: TestDefinition) -> Set[Type[ReportGenerationStrategy]]:
    return DEFAULT_REPORTERS.get(type(tdef), set())


def parse_time_limit(limit: str) -> timedelta:
    try:
        if re.match(r"^\d+[smhdw]$", limit, re.IGNORECASE):
            return parse_abbreviated_time(limit)
        if "-" in limit:
            return parse_dashed_time(limit)
        if len(limit.split(":")) == 3:
            hours, minutes, seconds = map(int, limit.split(":"))
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        if len(limit.split(":")) == 2:
            hours, minutes = map(int, limit.split(":"))
            return timedelta(hours=hours, minutes=minutes)
    except ValueError as err:
        raise ValueError(f"Invalid time limit format: {limit}. Refer to SLURM time format documentation.") from err

    raise ValueError(f"Unsupported time limit format: {limit}. Refer to SLURM time format documentation.")


def parse_abbreviated_time(limit: str) -> timedelta:
    value, unit = int(limit[:-1]), limit[-1].lower()
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Invalid abbreviated time format: {limit}")


def parse_dashed_time(limit: str) -> timedelta:
    days, time_part = limit.split("-", 1)
    hours, minutes, seconds = map(int, time_part.split(":"))
    return timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)


def format_time_limit(total_time: timedelta) -> str:
    total_seconds = int(total_time.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def calculate_total_time_limit(test_hooks: List[TestScenario], time_limit: Optional[str] = None) -> str:
    total_time = timedelta()

    if time_limit:
        total_time += parse_time_limit(time_limit)

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
        test_mapping: Dict[str, Test],
        hook_mapping: Dict[str, TestScenario],
        strict: bool = False,
    ) -> None:
        self.file_path = file_path
        self.system = system
        self.test_mapping = test_mapping
        self.hook_mapping = hook_mapping
        self.strict = strict

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
            tr.id: self._create_test_run(tr, normalized_weight, pre_test, post_test) for tr in ts_model.tests
        }

        tests_data: dict[str, TestRunModel] = {tr.id: tr for tr in ts_model.tests}
        for section, tr in test_runs_by_id.items():
            test_info = tests_data[section]
            tr.dependencies = {
                dep.type: TestDependency(test_run=test_runs_by_id[dep.id]) for dep in test_info.dependencies
            }

        return TestScenario(
            name=ts_model.name,
            test_runs=list(test_runs_by_id.values()),
            job_status_check=ts_model.job_status_check,
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
        original_test, tdef = self._prepare_tdef(test_info)

        test = Test(test_definition=tdef, test_template=original_test.test_template)

        hooks = [hook for hook in [pre_test, post_test] if hook is not None]
        total_time_limit = calculate_total_time_limit(test_hooks=hooks, time_limit=test_info.time_limit)

        tr = TestRun(
            test_info.id,
            test,
            num_nodes=test_info.num_nodes or 1,
            iterations=test_info.iterations,
            nodes=test_info.nodes,
            time_limit=total_time_limit,
            sol=test_info.sol,
            weight=test_info.weight * normalized_weight,
            ideal_perf=test_info.ideal_perf,
            pre_test=pre_test,
            post_test=post_test,
            reports=get_reporters(test_info, test.test_definition),
        )
        return tr

    def _prepare_tdef(self, test_info: TestRunModel) -> Tuple[Test, TestDefinition]:
        tp = TestParser([self.file_path], self.system)
        tp.current_file = self.file_path

        if test_info.test_name:
            if test_info.test_name not in self.test_mapping:
                raise ValueError(f"Test '{test_info.test_name}' is not defined. Was tests directory correctly set?")
            test = self.test_mapping[test_info.test_name]
        elif test_info.test_spec and test_info.test_spec.test_template_name:
            test = tp._parse_data(test_info.test_spec.model_dump(), self.strict)
        else:
            # this should never happen, because we check for this in the modelvalidator
            raise ValueError(f"Cannot configure test case '{test_info.id}' with both 'test_name' and 'test_spec'.")

        tdef = test.test_definition
        if test_info.test_spec:
            data = test.test_definition.model_dump()
            data.update(test_info.test_spec.model_dump(exclude_none=True, exclude_defaults=True))
            tdef = tp.load_test_definition(data, self.strict)

        return test, tdef
