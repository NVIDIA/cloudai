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

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cloudai.core import GitRepo, Installable, JobStatusResult, PythonExecutable, TestRun


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


class TestDefinition(BaseModel, ABC):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    extra_env_vars: dict[str, Union[str, List[str]]] = {}
    extra_cmd_args: dict[str, str] = {}
    extra_container_mounts: list[str] = []
    git_repos: list[GitRepo] = []
    nsys: Optional[NsysConfiguration] = None
    predictor: Optional[PredictorConfig] = None
    agent: str = "grid_search"
    agent_steps: int = 1
    agent_metrics: list[str] = Field(default=["default"])
    agent_reward_function: str = "inverse"
    agent_config: Optional[dict[str, Any]] = None

    @property
    def cmd_args_dict(self) -> Dict[str, Union[str, List[str]]]:
        return self.cmd_args.model_dump()

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k}={v}" if v else k)
        return " ".join(parts)

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos]

    def constraint_check(self, tr: TestRun) -> bool:
        return True

    @property
    def is_dse_job(self) -> bool:
        def check_dict(d: dict) -> bool:
            if isinstance(d, dict):
                for value in d.values():
                    if isinstance(value, list) or (isinstance(value, dict) and check_dict(value)):
                        return True
            return False

        return check_dict(self.cmd_args_dict) or check_dict(self.extra_env_vars)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)
