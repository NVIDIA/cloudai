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

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

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
            "persists to env.csv, and includes them in the trajectory cache key."
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
        def check_dict(d: dict, parent_key: str = "", skip_env_params: bool = False) -> bool:
            if isinstance(d, dict):
                for key, value in d.items():
                    path = f"{parent_key}.{key}" if parent_key else key
                    if self.is_dse_excluded_arg(path):
                        continue
                    if skip_env_params and self.is_env_sampled(path):
                        continue
                    if isinstance(value, list) or (
                        isinstance(value, dict) and check_dict(value, path, skip_env_params)
                    ):
                        return True
            return False

        return check_dict(self.cmd_args_dict, skip_env_params=True) or check_dict(self.extra_env_vars)

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

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)

    @field_validator("agent", mode="after")
    @staticmethod
    def validate_agent(agent: str) -> str:
        registry = Registry()
        if agent not in registry.agents_map:
            raise ValueError(
                f"Agent {agent} is not registered. Available agents are: {', '.join(registry.agents_map.keys())}"
            )
        return agent

    @model_validator(mode="after")
    def validate_agent_config(self) -> Self:
        if self.agent_config is not None:
            agent_class = Registry().agents_map[self.agent]
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
