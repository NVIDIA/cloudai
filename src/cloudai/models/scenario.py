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

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer, field_validator, model_validator

from cloudai.core import CmdArgs, GitRepo, NsysConfiguration, Registry, Reporter, TestRun
from cloudai.models.workload import TestDefinition


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
        if not allow_scenario_reports and issubclass(Registry().scenario_reports[name], Reporter):
            raise ValueError(f"Scenario level report '{name}' is not allowed here.")
        try:
            parsed[name] = report_cfg.model_validate(dict_cfg)
        except ValidationError as e:
            raise ValueError(
                f"Error validating report configuration '{name}' as {report_cfg.__class__.__name__}: {e}"
            ) from e
    return parsed


class TestRunDependencyModel(BaseModel):
    """Model for test dependency in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    type: Literal["end_post_comp", "start_post_init", "start_post_comp"]
    id: str


class TestRunModel(BaseModel):
    """Model for test run in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    test_name: Optional[str] = None
    num_nodes: int | list[int] | None = None
    nodes: list[str] = Field(default_factory=list)
    weight: int = 0
    iterations: int = 1
    sol: Optional[float] = None
    ideal_perf: float = 1.0
    time_limit: Optional[str] = None
    dependencies: list[TestRunDependencyModel] = Field(default_factory=list)

    # test definition fields
    name: Optional[str] = None
    description: Optional[str] = None
    test_template_name: Optional[str] = None
    cmd_args: Optional[CmdArgs] = None
    extra_env_vars: Optional[dict[str, str]] = None
    extra_container_mounts: Optional[list[str]] = None
    git_repos: Optional[list[GitRepo]] = None
    nsys: Optional[NsysConfiguration] = None
    agent: Optional[str] = None
    agent_steps: Optional[int] = None
    agent_metrics: list[str] = Field(default=["default"])

    def tdef_model_dump(self, by_alias: bool) -> dict:
        """Return a dictionary with non-None values that correspond to the test definition fields."""
        data = {
            "name": self.name,
            "description": self.description,
            "test_template_name": self.test_template_name,
            "agent": self.agent,
            "agent_steps": self.agent_steps,
            "agent_metrics": self.agent_metrics,
            "extra_container_mounts": self.extra_container_mounts,
            "extra_env_vars": self.extra_env_vars if self.extra_env_vars else None,
            "cmd_args": self.cmd_args.model_dump(by_alias=by_alias) if self.cmd_args else None,
            "git_repos": [repo.model_dump() for repo in self.git_repos] if self.git_repos else None,
            "nsys": self.nsys.model_dump() if self.nsys else None,
        }
        return {k: v for k, v in data.items() if v is not None}

    @model_validator(mode="after")
    def check_test_name_or_type_is_set(self):
        has_base = self.test_name is not None
        if not has_base and (self.test_template_name is None or self.name is None or self.description is None):
            raise ValueError(
                "When 'test_name' is not set, the following fields must be set: "
                "'test_template_name', 'name', 'description'."
            )

        if not self.test_name:
            if not self.test_template_name:
                raise ValueError("'test_template_name' must be set if 'test_name' is not set.")

            registry = Registry()
            if self.test_template_name not in registry.test_definitions_map:
                raise ValueError(
                    f"Test type '{self.test_template_name}' not found in the test definitions. "
                    f"Possible values are: {', '.join(registry.test_definitions_map.keys())}"
                )
        else:
            if self.test_template_name:
                raise ValueError("'test_template_name' must not be set if 'test_name' is set.")

        return self


class ReportConfig(BaseModel):
    """Model for report configuration in test scenario."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = False


class TestScenarioModel(BaseModel):
    """Model for test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    sol_path: Optional[str] = None
    job_status_check: bool = True
    tests: list[TestRunModel] = Field(alias="Tests", min_length=1)
    pre_test: Optional[str] = None
    post_test: Optional[str] = None

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


class TestRunDetails(BaseModel):
    """
    Model for test run dump.

    Used for storing a single test run with all fields set during command generation.
    """

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    name: str
    nnodes: int
    nodes: list[str] = []
    output_path: Path
    iterations: int
    current_iteration: int
    step: int
    test_cmd: str
    full_cmd: str
    test_definition: Any

    @field_serializer("output_path")
    def _path_serializer(self, v: Path) -> str:
        return str(v.absolute())

    @field_validator("test_definition", mode="before")
    @classmethod
    def load_tdef(cls, value: Any | None) -> TestDefinition | None:
        if value is None:
            return None
        if isinstance(value, TestDefinition):
            return value
        if isinstance(value, dict):
            workload_type = value["test_template_name"]
            workload_cls = Registry().test_definitions_map[workload_type]
            return workload_cls.model_validate(value)

        raise ValueError(f"Invalid test definition: {value}")

    @classmethod
    def from_test_run(cls, tr: TestRun, test_cmd: str, full_cmd: str) -> TestRunDetails:
        return cls(
            name=tr.name,
            nnodes=tr.nnodes,
            nodes=tr.nodes,
            output_path=tr.output_path,
            iterations=tr.iterations,
            current_iteration=tr.current_iteration,
            step=tr.step,
            test_cmd=test_cmd,
            full_cmd=full_cmd,
            test_definition=tr.test.test_definition,
        )
