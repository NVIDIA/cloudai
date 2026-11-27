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

import copy
from abc import ABC
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

import cloudai.metrics
from cloudai.core import GitRepo, Installable, JobStatusResult, PythonExecutable, Registry, System, TestRun

from ..configurator.env_params import EnvParamSpec


class CmdArgs(BaseModel):
    """Test command arguments."""

    model_config = ConfigDict(extra="allow")


class NsysConfiguration(BaseModel):
    """NSYS configuration."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    nsys_binary: str = "nsys"
    task: str = "profile"
    output: Optional[str] = None
    sample: Optional[str] = None
    trace: Optional[str] = None
    force_overwrite: Optional[bool] = None
    capture_range: Optional[str] = None
    capture_range_end: Optional[str] = None
    cuda_graph_trace: Optional[str] = None
    gpu_metrics_devices: Optional[str] = None
    extra_args: list[str] = []

    @property
    def cmd_args(self) -> list[str]:
        parts = [f"{self.nsys_binary}", f"{self.task}"]
        if self.sample:
            parts.append(f"-s {self.sample}")
        if self.output:
            parts.append(f"-o {self.output}")
        if self.trace:
            parts.append(f"-t {self.trace}")
        if self.force_overwrite is not None:
            parts.append(f"--force-overwrite={str(self.force_overwrite).lower()}")
        if self.capture_range:
            parts.append(f"--capture-range={self.capture_range}")
        if self.capture_range_end:
            parts.append(f"--capture-range-end={self.capture_range_end}")
        if self.cuda_graph_trace:
            parts.append(f"--cuda-graph-trace={self.cuda_graph_trace}")
        if self.gpu_metrics_devices:
            parts.append(f"--gpu-metrics-devices={self.gpu_metrics_devices}")
        parts.extend(self.extra_args)

        return parts


@dataclass
class PredictorConfig(PythonExecutable):
    """Predictor configuration."""

    bin_name: Optional[str] = None

    def __hash__(self) -> int:
        """
        Hash the PredictorConfig.

        It is based on git repo on purpose to avoid re-downloading the same repo for multiple scripts.
        """
        return self.git_repo.__hash__()


class TrainingReportConfig(BaseModel):
    """Training-report aggregation window: steps excluded before computing per-metric stats."""

    model_config = ConfigDict(extra="forbid")

    exclude_start_steps: int = Field(default=5, ge=0)
    exclude_post_profiling_steps: int = Field(default=2, ge=0)


class TestDefinition(BaseModel, ABC):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    dse_excluded_args: list[str] = Field(default_factory=list)
    extra_env_vars: dict[str, Union[str, List[str]]] = {}
    extra_cmd_args: dict[str, str] = {}
    extra_container_mounts: list[str] = []
    git_repos: list[GitRepo] = []
    nsys: Optional[NsysConfiguration] = None
    predictor: Optional[PredictorConfig] = None
    training_report: Optional[TrainingReportConfig] = None

    agent: str = "grid_search"
    agent_steps: int = 1
    agent_metrics: list[str] = Field(default=["default"])
    agent_reward_function: str = "inverse"
    agent_config: dict[str, Any] | None = Field(default=None, description="Agent configuration.")
    env_params: dict[str, EnvParamSpec] = Field(
        default_factory=dict,
        description=(
            "Environment parameters sampled by the env per trial. Sibling to "
            "cmd_args; not part of the agent's action space. CloudAIGymEnv samples, "
            "persists them in the trajectory output, and includes them in the trajectory cache key."
        ),
    )

    @property
    def cmd_args_dict(self) -> Dict[str, Union[str, List[str]]]:
        return self.cmd_args.model_dump()

    def is_dse_excluded_arg(self, path: str) -> bool:
        """Return whether a dot-separated cmd_args path should be ignored by DSE."""
        path = f"cmd_args.{path}"
        return any(path == excluded or path.startswith(f"{excluded}.") for excluded in self.dse_excluded_args)

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k}={v}" if v else k)
        return " ".join(parts)

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos]

    def constraint_check(self, tr: TestRun, system: Optional[System]) -> bool:
        return True

    def is_env_sampled(self, cmd_args_path: str) -> bool:
        """Whether a cmd_args field is env-sampled (env draws it per trial, not the agent)."""
        return cmd_args_path in self.env_params

    @property
    def is_domain_randomization_enabled(self) -> bool:
        """Whether the config declares domain randomization: at least one ``env_params`` annotation."""
        return bool(self.env_params)

    @property
    def is_dse_job(self) -> bool:
        def check_dict(
            d: dict, parent_key: str = "", skip_env_params: bool = False, sweepable_only: bool = False
        ) -> bool:
            if isinstance(d, dict):
                for key, value in d.items():
                    path = f"{parent_key}.{key}" if parent_key else key
                    if self.is_dse_excluded_arg(path):
                        continue
                    if skip_env_params and self.is_env_sampled(path):
                        continue
                    if isinstance(value, list) and (not sweepable_only or self.is_sweepable_cmd_arg(path)):
                        return True
                    if isinstance(value, dict) and check_dict(value, path, skip_env_params, sweepable_only):
                        return True
            return False

        return check_dict(self.cmd_args_dict, skip_env_params=True, sweepable_only=True) or check_dict(
            self.extra_env_vars
        )

    def is_sweepable_cmd_arg(self, path: str) -> bool:
        """Return whether a cmd_args path can represent either a scalar or sweep list."""
        value: Any = self.cmd_args
        annotation: Any = Any
        for part in path.split("."):
            if isinstance(value, BaseModel):
                field = type(value).model_fields.get(part)
                if field is not None:
                    annotation = field.annotation
                    value = getattr(value, part)
                    continue
                extra = getattr(value, "__pydantic_extra__", None)
                if extra is None or part not in extra:
                    return False
                annotation = Any
                value = extra[part]
                continue

            if isinstance(value, dict):
                mapping_annotation = annotation
                origin = get_origin(mapping_annotation)
                if origin in (Union, UnionType):
                    mapping_annotation = next(
                        (arg for arg in get_args(mapping_annotation) if get_origin(arg) is dict or arg is dict), Any
                    )
                value_args = get_args(mapping_annotation)
                annotation = value_args[1] if len(value_args) == 2 else Any
                value = value.get(part)
                continue

            return False

        return annotation is Any or self._annotation_admits_scalar(annotation)

    @field_validator("dse_excluded_args", mode="before")
    @classmethod
    def normalize_dse_excluded_args(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]

        normalized = []
        for prefix in value:
            if not isinstance(prefix, str):
                raise ValueError("DSE excluded cmd_args prefixes must be strings.")

            prefix = prefix.strip()
            if not prefix.startswith("cmd_args."):
                raise ValueError(f"DSE excluded arg must start with 'cmd_args.': {prefix!r}")
            if prefix == "cmd_args." or prefix.endswith(".") or ".." in prefix:
                raise ValueError(f"Invalid DSE excluded cmd_args prefix: {prefix!r}")

            normalized.append(prefix)

        return normalized

    @staticmethod
    def _annotation_admits_scalar(annotation: Any) -> bool:
        """Return whether an annotation explicitly accepts one of the scalar sweep values."""
        origin = get_origin(annotation)
        if origin is Annotated:
            return TestDefinition._annotation_admits_scalar(get_args(annotation)[0])
        if origin is Literal:
            return bool(get_args(annotation)) and all(
                isinstance(value, (int, float, str, bool)) for value in get_args(annotation)
            )
        if origin in (Union, UnionType):
            return any(TestDefinition._annotation_admits_scalar(arg) for arg in get_args(annotation))
        return annotation in (int, float, str, bool)

    @staticmethod
    def _annotation_accepts_scalar(value: Any, annotation: Any) -> bool:
        """Return whether a scalar value satisfies a scalar alternative in an annotation."""
        if annotation is Any:
            return True
        origin = get_origin(annotation)
        if origin is Annotated:
            return TestDefinition._annotation_accepts_scalar(value, get_args(annotation)[0])
        if origin is Literal:
            return any(type(value) is type(candidate) and value == candidate for candidate in get_args(annotation))
        if origin in (Union, UnionType):
            return any(TestDefinition._annotation_accepts_scalar(value, arg) for arg in get_args(annotation))
        return annotation in (int, float, str, bool) and isinstance(value, annotation)

    def _canonicalize_sweep_value(
        self,
        value: Any,
        annotation: Any,
        path: str,
        *,
        cmd_args: bool,
        untyped_extra: bool = False,
    ) -> Any:
        """Collapse singleton scalar candidates, following the declared value type."""
        if cmd_args and (self.is_env_sampled(path) or self.is_dse_excluded_arg(path)):
            return value

        if isinstance(value, BaseModel):
            self._canonicalize_sweep_model(value, path, cmd_args=cmd_args)
            return value

        if isinstance(value, dict):
            mapping_annotation = annotation
            origin = get_origin(mapping_annotation)
            if origin in (Union, UnionType):
                mapping_annotation = next(
                    (arg for arg in get_args(mapping_annotation) if get_origin(arg) is dict or arg is dict), Any
                )
            value_args = get_args(mapping_annotation)
            value_annotation = value_args[1] if len(value_args) == 2 else Any
            for key, nested_value in list(value.items()):
                nested_path = f"{path}.{key}" if path else str(key)
                value[key] = self._canonicalize_sweep_value(
                    nested_value,
                    value_annotation,
                    nested_path,
                    cmd_args=cmd_args,
                    untyped_extra=untyped_extra or value_annotation is Any,
                )
            return value

        if (
            isinstance(value, list)
            and len(value) == 1
            and self._is_scalar_sweep_value(value[0])
            and (untyped_extra or self._annotation_accepts_scalar(value[0], annotation))
        ):
            return value[0]

        return value

    @staticmethod
    def _is_scalar_sweep_value(value: Any) -> bool:
        return isinstance(value, (int, float, str, bool))

    def _canonicalize_sweep_model(self, model: BaseModel, parent_path: str, *, cmd_args: bool) -> None:
        """Walk declared and extra fields in a cmd_args model."""
        for name, field in type(model).model_fields.items():
            path = f"{parent_path}.{name}" if parent_path else name
            value = getattr(model, name)
            normalized = self._canonicalize_sweep_value(value, field.annotation, path, cmd_args=cmd_args)
            if normalized is not value:
                setattr(model, name, normalized)

        extra = getattr(model, "__pydantic_extra__", None)
        if extra:
            for name, value in list(extra.items()):
                path = f"{parent_path}.{name}" if parent_path else name
                normalized = self._canonicalize_sweep_value(value, Any, path, cmd_args=cmd_args, untyped_extra=True)
                if normalized is not value:
                    extra[name] = normalized

    @model_validator(mode="after")
    def canonicalize_singleton_sweeps(self) -> Self:
        """Store fixed scalar choices as scalars, leaving multi-value sweeps unchanged."""
        if isinstance(self.cmd_args, BaseModel):
            self.cmd_args = self.cmd_args.model_copy(deep=True)
            self._canonicalize_sweep_model(self.cmd_args, "", cmd_args=True)

        extra_env_field = type(self).model_fields["extra_env_vars"]
        self.extra_env_vars = self._canonicalize_sweep_value(
            copy.deepcopy(self.extra_env_vars),
            extra_env_field.annotation,
            "",
            cmd_args=False,
        )
        return self

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        """Return canonical metric observations produced by this test run, if supported."""
        return []

    @field_validator("agent", mode="after")
    @staticmethod
    def validate_agent(agent: str) -> str:
        registry = Registry()
        if not registry.has_agent(agent):
            raise ValueError(
                f"Agent {agent} is not registered. Available agents are: {', '.join(registry.agent_names())}"
            )
        return agent

    @model_validator(mode="after")
    def validate_agent_config(self) -> Self:
        if self.agent_config is not None:
            agent_class = Registry().get_agent(self.agent)
            agent_config_class = agent_class.get_config_class()
            agent_config_class.model_validate(self.agent_config)
        return self

    @model_validator(mode="after")
    def validate_env_params(self) -> Self:
        """
        Validate env_params annotations against cmd_args.

        ``env_params`` is an annotation: each key names a ``cmd_args`` field whose value is
        the candidate set (the single source of truth), and the entry carries only *how* to
        sample. So each key must name a real ``cmd_args`` field whose value is a candidate
        list; a scalar is already fixed, so annotating it is a meaningless label and is
        rejected here. When ``weights`` are declared, the list needs >= 2 values and the
        weights must align 1:1 with it. Sampling, persistence, the per-trial cmd_args overlay,
        and the cache key all
        live in ``CloudAIGymEnv``; keeping this shape check in core lets the overlay stay
        agent- and workload-agnostic rather than re-implemented per workload.
        """
        if not self.env_params:
            return self

        cmd_args_fields = getattr(type(self.cmd_args), "model_fields", None)
        if not cmd_args_fields:
            return self

        unknown = sorted(k for k in self.env_params if k not in cmd_args_fields)
        if unknown:
            raise ValueError(f"env_params keys {unknown} are not cmd_args fields on {type(self.cmd_args).__name__}")

        for name, spec in self.env_params.items():
            self._validate_env_param_field(name, spec, getattr(self.cmd_args, name, None))
        return self

    @staticmethod
    def _validate_env_param_field(name: str, spec: Any, value: Any) -> None:
        """Reject one env_params entry whose target cmd_args field is not a valid candidate list."""
        if isinstance(value, (dict, BaseModel)):
            raise ValueError(
                f"env_params['{name}'] must target a leaf cmd_args field (a candidate list), "
                "not a structured object; param_space/is_dse_job exclude the whole key, which would "
                "silently drop nested action dimensions"
            )
        if not isinstance(value, list):
            raise ValueError(
                f"env_params['{name}'] annotates cmd_args.{name}, which is not a candidate list "
                f"(got {type(value).__name__}); the annotation only reclassifies a list-valued sweep as "
                f"env-sampled, while a scalar is already fixed. Make cmd_args.{name} a list or remove "
                "the annotation"
            )
        if not value:
            raise ValueError(
                f"env_params['{name}'] references an empty candidate list in cmd_args.{name}; "
                "provide at least one candidate (the sampler would otherwise fail on an empty draw)"
            )
        if len(value) < 2:
            raise ValueError(
                f"env_params['{name}'] needs >= 2 candidate values in cmd_args.{name}; "
                "a single-element list is a fixed value, not domain randomization"
            )
        if spec.weights is None:
            return
        if len(spec.weights) != len(value):
            raise ValueError(
                f"env_params['{name}'] weights length {len(spec.weights)} does not match "
                f"cmd_args.{name} candidate count {len(value)}"
            )
