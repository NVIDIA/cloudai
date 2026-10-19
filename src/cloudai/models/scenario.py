# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    ValidationError,
    field_validator,
    model_validator,
)

import cloudai.metrics
from cloudai._core.registry import Registry
from cloudai.models.workload import TestDefinition

if TYPE_CHECKING:
    from cloudai._core.test_run import TestRun


def parse_reports_spec(
    value: dict[str, Any] | None, allow_scenario_reports: bool = True
) -> dict[str, ReportConfig] | None:
    if value is None:
        return None

    parsed: dict[str, ReportConfig] = {}
    for name, dict_cfg in value.items():
        report_cfg = Registry().report_configs.get(name)
        if not report_cfg:
            raise ValueError(
                f"Report configuration for '{name}' not found in the registry. "
                f"Available reports: {', '.join(Registry().report_configs.keys())}"
            )
        if not allow_scenario_reports and name in Registry().scenario_reports:
            raise ValueError(f"Scenario level report '{name}' is not allowed here.")
        try:
            parsed[name] = report_cfg.model_validate(dict_cfg)
        except ValidationError as e:
            raise ValueError(
                f"Error validating report configuration '{name}' as {report_cfg.__class__.__name__}: {e}"
            ) from e
    return parsed


class ReportConfig(BaseModel):
    """Model for report configuration in test scenario."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = False


@dataclass(frozen=True)
class ConfigPaths:
    """Source paths used to build a system and test scenario."""

    system_path: Path
    tests_dir_path: Path | None
    test_scenario_path: Path


class TestScenario(BaseModel):
    """Scenario configuration and its temporary bindings to the existing runner."""

    __test__ = False
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, populate_by_name=True)

    name: str
    tests: list[SerializeAsAny[TestDefinition]] = Field(alias="Tests")
    sol: cloudai.metrics.MetricSOLConfig = Field(default_factory=dict)
    job_status_check: bool = True
    pre_test: str | None = None
    post_test: str | None = None
    reports: dict[str, SerializeAsAny[ReportConfig]] = Field(default_factory=dict)
    config_paths: ConfigPaths | None = None
    _test_runs: list[TestRun] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        # Runners and workload hooks still construct scenarios from execution objects.
        runs = data.pop("test_runs", None)
        if runs is not None and "tests" not in data and "Tests" not in data:
            # Multiple bound trials can share one configured scenario case.
            data["tests"] = list(
                {run.name: run.test.model_copy(deep=True, update={"id": run.name}) for run in runs}.values()
            )
        super().__init__(**data)
        if runs is not None:
            self._test_runs = runs

    @property
    def test_runs(self) -> list[TestRun]:
        """Return bound executions; replacing this list never changes configuration."""
        return self._test_runs

    @test_runs.setter
    def test_runs(self, value: list[TestRun]) -> None:
        self._test_runs = value

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
            if tr.id is None:
                raise ValueError(f"Test '{tr.name}' requires a scenario id.")
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

    @field_validator("sol", mode="before")
    @classmethod
    def parse_sol(cls, value: dict[str, Any] | None) -> cloudai.metrics.MetricSOLConfig:
        return cloudai.metrics.parse_sol_spec(value)

    @field_validator("reports", mode="before")
    @classmethod
    def parse_reports(cls, value: dict[str, Any] | None) -> dict[str, ReportConfig] | None:
        return parse_reports_spec(value)

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


# Temporary compatibility for internal projects, including CloudAIX.
# Remove once their writers serialize TestRun directly.
class TestRunDetails:
    """Deprecated wrapper retained only for internal project compatibility."""

    @staticmethod
    def from_test_run(tr: TestRun, test_cmd: str = "", full_cmd: str = "") -> TestRun:
        """Return the run itself; command strings are no longer saved."""
        warnings.warn(
            "TestRunDetails is retained only for internal project compatibility and will be removed. "
            'Serialize TestRun directly with model_dump(mode="json") instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return tr
