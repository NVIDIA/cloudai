# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cloudai.metrics
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
from .models.workload import TestDefinition
from .test_parser import TestParser, load_test_toml_file
from .toml_utils import format_toml_decode_error
from .workloads.nccl_test import NCCLTestDefinition, NcclTestPredictionReportGenerationStrategy


def get_reporters(tdef: TestDefinition) -> Set[Type[ReportGenerationStrategy]]:
    reporters = copy.deepcopy(Registry().reports_map.get(type(tdef), set()))

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
            data = load_scenario_toml_file(file, self.file_path)
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
            resolved = dict(data)
            items = data.get("Tests")
            if not isinstance(items, list) or not items:
                raise ValueError("Tests must contain at least one test definition")
            resolved["Tests"] = [self._prepare_tdef(item) for item in items]
            ts_model = TestScenario.model_validate(resolved)
        except (ValidationError, ValueError) as e:
            msg = f"Failed to parse Test Scenario definition: {self.file_path}"
            if isinstance(e, ValidationError):
                for err in e.errors(include_url=False):
                    msg += f"\n\t{format_validation_error(err)}"
            else:
                msg += f"\n\t{e}"
            logging.error(msg)
            raise TestScenarioParsingError(msg) from e

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
            (trm.id or trm.name): self._create_test_run(trm, normalized_weight, pre_test, post_test, ts_model.sol)
            for trm in ts_model.tests
        }

        for test_info in ts_model.tests:
            tr = test_runs_by_id[test_info.id or test_info.name]
            tr.dependencies = {
                dep.type: TestDependency(test_run=test_runs_by_id[dep.id]) for dep in test_info.dependencies
            }

        ts_model.test_runs = list(test_runs_by_id.values())
        return ts_model

    def _create_test_run(
        self,
        test_info: TestDefinition,
        normalized_weight: float,
        pre_test: Optional[TestScenario] = None,
        post_test: Optional[TestScenario] = None,
        scenario_sol: cloudai.metrics.SOLMetricModel | None = None,
    ) -> TestRun:
        """
        Prepare effective configuration and bind it to a new execution.

        Args:
            test_info: Original scenario definition, kept unchanged.
            normalized_weight (float): Normalized weight for the test.
            pre_test (Optional[TestScenario]): TestScenario object representing the pre-test sequence.
            post_test (Optional[TestScenario]): TestScenario object representing the post-test sequence.
            scenario_sol: Scenario-level metric SOL configuration.

        Returns:
            TestRun: Execution bound to its prepared definition.

        Raises:
            ValueError: If the test or nodes are not found within the system.
        """
        hooks = [hook for hook in [pre_test, post_test] if hook is not None]
        total_time_limit = calculate_total_time_limit(test_hooks=hooks, time_limit=test_info.time_limit)

        sol = test_info.sol
        if not isinstance(sol, (float, int)):
            targets = cloudai.metrics.merge_sol_configs(self.system.sol, scenario_sol, sol)
            sol = cloudai.metrics.SOLMetricModel(targets) if targets else None
        num_nodes = test_info.num_nodes
        if num_nodes is not None and not num_nodes:
            num_nodes = 1

        prepared = test_info.model_copy(
            deep=True,
            update={
                "num_nodes": num_nodes,
                "time_limit": total_time_limit,
                "sol": sol,
                "weight": test_info.weight * normalized_weight,
                "reporters": get_reporters(test_info),
            },
        )
        return TestRun(test=prepared, pre_test=pre_test, post_test=post_test)

    def _prepare_tdef(self, data: dict[str, Any]) -> TestDefinition:  # noqa: C901
        """Resolve a scenario item into its sole, workload-specific definition model."""
        if not isinstance(data, dict):
            raise ValueError("Each Tests item must be a table")

        overrides = {key: value for key, value in data.items() if value is not None}

        # Empty overrides preserve values from the referenced test
        # test: [extra_env_vars] A=B
        # scenario case: [extra_env_vars]
        # resolved scenario case: [extra_env_vars] A=B
        if overrides.get("extra_env_vars") == {}:
            overrides.pop("extra_env_vars")
        if overrides.get("git_repos") == []:
            overrides.pop("git_repos")

        test_name = overrides.pop("test_name", None)
        path = overrides.pop("path", None)
        if test_name is not None and path is not None:
            raise ValueError("'test_name' and 'path' must not both be set")
        if (test_name is not None or path is not None) and "test_template_name" in overrides:
            raise ValueError("'test_template_name' must not be set with 'test_name' or 'path'")
        source = self.file_path

        if test_name is not None:
            try:
                base = self.test_mapping[test_name]
            except (KeyError, TypeError) as e:
                raise MissingTestError(str(test_name)) from e
            test_defined = base.model_dump(by_alias=True, exclude_unset=True)
        elif path is not None:
            if not isinstance(path, str) or not path:
                raise ValueError("'path' must be a non-empty string")
            source = (self.file_path.parent / path).resolve()
            if not source.is_file():
                msg = (
                    f"Test case '{data.get('id')}' references path '{path}', "
                    f"which resolves to '{source}', but that file does not exist."
                )
                logging.error(msg)
                raise TestScenarioParsingError(msg)

            with source.open() as fh:
                test_defined = load_test_toml_file(fh, source)
        else:
            test_defined = {}

        parser = TestParser([], self.system)
        parser.current_file = source
        return parser.load_test_definition(deep_merge(test_defined, overrides))


def load_scenario_toml_file(fh, file_path: Path) -> Dict[str, Any]:
    try:
        return toml.load(fh)
    except toml.TomlDecodeError as e:
        message = format_toml_decode_error(file_path, e, "test scenario")
        logging.error(message)
        raise TestScenarioParsingError(message) from e
