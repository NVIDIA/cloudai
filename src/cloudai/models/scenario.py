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

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

import cloudai.metrics
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


SCENARIO_TEST_RUN_FIELDS: set[str] = {
    "id",
    "test_name",
    "path",
    "num_nodes",
    "nodes",
    "pin_nodes",
    "exclude_nodes",
    "weight",
    "iterations",
    "sol",
    "ideal_perf",
    "time_limit",
    "dependencies",
    "extra_srun_args",
}


class TestRunModel(BaseModel):
    """Model for test run in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    test_name: Optional[str] = None
    path: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Path to a test TOML file, resolved relative to this scenario file's own directory. "
            "Alternative to 'test_name': references a test by file location instead of by name."
        ),
    )
    num_nodes: int | list[int] | None = None
    nodes: list[str] = Field(default_factory=list)
    pin_nodes: bool = False
    exclude_nodes: list[str] = Field(
        default_factory=list,
        description=(
            "Hostnames to exclude from the resolved node list. "
            "Supports Slurm range syntax, e.g. ['node-048', 'node-[101-104]']."
        ),
    )
    weight: int = 0
    iterations: int = 1
    sol: float | cloudai.metrics.MetricSOLConfig | None = None
    ideal_perf: float = 1.0
    time_limit: Optional[str] = None
    dependencies: list[TestRunDependencyModel] = Field(default_factory=list)
    extra_srun_args: str | None = None

    # test definition fields
    name: Optional[str] = None
    description: Optional[str] = None
    test_template_name: Optional[str] = None
    cmd_args: Optional[CmdArgs] = None
    dse_excluded_args: Optional[list[str]] = None
    extra_env_vars: dict[str, str | list[str]] | None = None
    extra_container_mounts: Optional[list[str]] = None
    git_repos: Optional[list[GitRepo]] = None
    nsys: Optional[NsysConfiguration] = None
    agent: Optional[str] = None
    agent_steps: Optional[int] = None
    agent_metrics: list[str] = Field(default=["default"])
    agent_reward_function: Optional[str] = None
    agent_config: Optional[dict[str, Any]] = Field(default=None, description="Agent configuration.")

    @field_validator("sol", mode="before")
    @classmethod
    def parse_sol(cls, value: Any) -> float | cloudai.metrics.MetricSOLConfig | None:
        if isinstance(value, dict):
            return cloudai.metrics.parse_sol_spec(value)
        return value

    @model_validator(mode="after")
    def validate_pin_nodes(self) -> Self:
        if self.pin_nodes and isinstance(self.num_nodes, list):
            raise ValueError("pin_nodes cannot be enabled when num_nodes is swept")
        return self

    def tdef_model_dump(self, by_alias: bool) -> dict:
        """Return explicitly set workload fields, including workload-specific fields."""
        data = self.model_dump(by_alias=by_alias, exclude=SCENARIO_TEST_RUN_FIELDS, exclude_unset=True)
        # Empty values for these fields did not replace values in referenced tests.
        for field in ("extra_env_vars", "git_repos"):
            if not data.get(field):
                data.pop(field, None)
        return {k: v for k, v in data.items() if v is not None}

    @model_validator(mode="after")
    def check_test_name_or_type_is_set(self):
        if self.test_name is not None and self.path is not None:
            raise ValueError("'test_name' and 'path' must not both be set; use only one to reference a test.")

        has_base = self.test_name is not None or self.path is not None
        if not has_base and (self.test_template_name is None or self.name is None or self.description is None):
            raise ValueError(
                "When neither 'test_name' nor 'path' is set, the following fields must be set: "
                "'test_template_name', 'name', 'description'."
            )

        if not has_base:
            if not self.test_template_name:
                raise ValueError("'test_template_name' must be set if 'test_name' is not set.")

            registry = Registry()
            if self.test_template_name not in registry.test_definitions_map:
                raise ValueError(
                    f"Test type '{self.test_template_name}' not found in the test definitions. "
                    f"Possible values are: {', '.join(registry.test_definitions_map.keys())}"
                )
        elif self.test_template_name is not None:
            raise ValueError("'test_template_name' must not be set if 'test_name' or 'path' is set.")

        return self

    @field_validator("agent", mode="after")
    @staticmethod
    def validate_agent(agent: str | None) -> str | None:
        registry = Registry()
        if agent is not None and not registry.has_agent(agent):
            raise ValueError(
                f"Agent {agent} is not registered. Available agents are: {', '.join(registry.agent_names())}"
            )
        return agent

    @model_validator(mode="after")
    def validate_agent_config(self) -> Self:
        if self.agent is not None and self.agent_config is not None:
            agent_class = Registry().get_agent(self.agent)
            agent_config_class = agent_class.get_config_class()
            agent_config_class.model_validate(self.agent_config)
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
    sol: cloudai.metrics.MetricSOLConfig = Field(default_factory=dict)
    job_status_check: bool = True
    tests: list[TestRunModel] = Field(alias="Tests", min_length=1)
    pre_test: Optional[str] = None
    post_test: Optional[str] = None
    reports: dict[str, ReportConfig] = Field(default_factory=dict)

    @field_validator("sol", mode="before")
    @classmethod
    def parse_sol(cls, value: dict[str, Any] | None) -> cloudai.metrics.MetricSOLConfig:
        return cloudai.metrics.parse_sol_spec(value)

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

    @field_validator("reports", mode="before")
    @classmethod
    def parse_reports(cls, value: dict[str, Any] | None) -> dict[str, ReportConfig] | None:
        return parse_reports_spec(value)


def _export_comparison_view(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _export_comparison_view(child) for key, child in value.items() if child is not None}
    if isinstance(value, list):
        return [_export_comparison_view(child) for child in value]
    return value


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
    def load_tdef(cls, value: Any | None, info: ValidationInfo) -> TestDefinition | None:
        if value is None:
            return None
        if isinstance(value, TestDefinition):
            definition = value
        elif isinstance(value, dict):
            workload_type = value["test_template_name"]
            workload_cls = Registry().test_definitions_map[workload_type]
            definition = workload_cls.model_validate(value)
        else:
            raise ValueError(f"Invalid test definition: {value}")
        if not isinstance(info.context, dict) or "trajectory_record" not in info.context:
            return definition

        import json
        from copy import deepcopy

        from pydantic import TypeAdapter

        from cloudai.configurator.trajectory import (
            TrajectoryLoadError,
            _merge_recorded_values,
            _record_environment,
            _resolve_recorded_action,
        )

        record = info.context["trajectory_record"]
        try:
            details = record.get("run_details", {}, decode_numbers=True)
            projection = {
                "purpose": "trajectory",
                "record": {
                    "supplied": deepcopy(info.context["recorded_dump"]),
                    "recorded": {
                        "environment": record.get("env_params", {}, decode_numbers=True),
                        "action": details.get("resolved_action", record["action"]),
                        "details": details,
                    },
                },
            }
            projection["record"] = TypeAdapter(dict[str, Any]).dump_python(projection["record"], mode="json")
            definition.model_dump(mode="json", exclude_unset=True, context=projection)
            supplied = projection["record"]["supplied"]
            recorded = projection["record"]["recorded"]
            native = record.get("record_context_version") == "1" and "recovery" not in details
            action = recorded["action"]
            resources = {"NUM_NODES": supplied["nnodes"]} if "nnodes" in supplied else {}
            configuration = supplied["test_definition"]
            resolved = _resolve_recorded_action(configuration, resources, action, missing_ok=True)
            action = {name: _export_comparison_view(value) for name, value in action.items()}
            resolved = {name: _export_comparison_view(value) for name, value in resolved.items()}
            if native and json.dumps(action, sort_keys=True, allow_nan=False) != json.dumps(
                resolved, sort_keys=True, allow_nan=False
            ):
                raise ValueError("native winning action cannot be verified from selected dump")
            _merge_recorded_values(action, resolved, action, record, typed=True)
            environment = _export_comparison_view(_record_environment(deepcopy(configuration), resources, action))
            recorded["environment"] = _export_comparison_view(recorded["environment"])
            merged = _merge_recorded_values(
                recorded["environment"], environment, recorded["environment"], record, typed=True
            )
            if native and json.dumps(merged, sort_keys=True) != json.dumps(recorded["environment"], sort_keys=True):
                raise ValueError("selected dump introduces unverified native settings")
            recorded_details = recorded["details"]
            identity = {
                key: recorded_details[name]
                for key, name in {
                    "name": "run_name",
                    "current_iteration": "iteration",
                    "step": "step",
                    "output_path": "output_path",
                }.items()
                if name in recorded_details
            }
            identity.update(
                {
                    key: recorded_details.get("launch", {})[name]
                    for key, name in {
                        "nnodes": "num_nodes",
                        "nodes": "nodes",
                        "exclude_nodes": "exclude_nodes",
                        "pin_nodes": "pin_nodes",
                        "time_limit": "time_limit",
                        "extra_srun_args": "extra_srun_args",
                    }.items()
                    if name in recorded_details.get("launch", {})
                }
            )
            if (
                "output_path" in identity
                and "output_path" in supplied
                and Path(identity["output_path"]).is_absolute() != Path(supplied["output_path"]).is_absolute()
            ):
                identity.pop("output_path")
            identity = {name: _export_comparison_view(value) for name, value in identity.items()}
            _merge_recorded_values(
                identity,
                {name: _export_comparison_view(supplied[name]) for name in identity if name in supplied},
                identity,
                record,
                typed=True,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise TrajectoryLoadError(
                record.source, "selected export dump conflicts with recorded configuration", step=record.step
            ) from error
        return definition

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
            test_definition=tr.test,
        )
