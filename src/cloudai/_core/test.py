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

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .installables import GitRepo, Installable, PythonExecutable
from .test_template import TestTemplate


class Test:
    """Represent a test, an instance of a test template with custom arguments, node configuration, and other details."""

    __test__ = False

    def __init__(self, test_definition: "TestDefinition", test_template: TestTemplate) -> None:
        """
        Initialize a Test instance.

        Args:
            test_definition (TestDefinition): The test definition object.
            test_template (TestTemplate): The test template object
        """
        self.test_template = test_template
        self.test_definition = test_definition

    def __repr__(self) -> str:
        """
        Return a string representation of the Test instance.

        Returns
            str: String representation of the test.
        """
        return (
            f"Test(name={self.name}, description={self.description}, "
            f"test_template={self.test_template.name}, "
            f"cmd_args={self.cmd_args}, "
            f"extra_env_vars={self.extra_env_vars}, "
            f"extra_cmd_args={self.extra_cmd_args}"
        )

    @property
    def name(self) -> str:
        return self.test_definition.name

    @property
    def description(self) -> str:
        return self.test_definition.description

    @property
    def cmd_args(self) -> Dict[str, Union[str, List[str]]]:
        return self.test_definition.cmd_args_dict

    @property
    def extra_cmd_args(self) -> str:
        return self.test_definition.extra_args_str

    @property
    def extra_env_vars(self) -> Dict[str, Union[str, List[str]]]:
        return self.test_definition.extra_env_vars


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
        return self.git_repo.__hash__()


class TestDefinition(BaseModel, ABC):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    extra_env_vars: Dict[str, Union[str, List[str]]] = {}
    extra_cmd_args: dict[str, str] = {}
    extra_container_mounts: list[str] = []
    git_repos: list[GitRepo] = []
    nsys: Optional[NsysConfiguration] = None
    predictor: Optional[PredictorConfig] = None
    agent: str = "grid_search"
    agent_steps: int = 1
    agent_metric: str = "default"

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
        return []

    @property
    def constraint_check(self) -> bool:
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
