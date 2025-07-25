# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, FieldValidationInfo

from cloudai.core import DockerImage, Installable, File
from cloudai.models.workload import CmdArgs, TestDefinition


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: Union[int, list[int]]
    extra_args: str = ""

class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for prefill worker."""
    pass

class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for decode worker."""
    pass

class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="allow")

    prefill_worker: PrefillWorkerArgs
    decode_worker: DecodeWorkerArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    extra_args: str = ""



class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    huggingface_home_host_path: Path = Path.home() / ".cache/huggingface"
    huggingface_home_container_path: Path = Path("/root/.cache/huggingface")
    skip_huggingface_home_host_path_validation: bool = False
    dynamo: AIDynamoArgs
    sleep_seconds: int = 660
    genai_perf: GenAIPerfArgs
    node_setup_cmd: str = ""
    extra_args: str = ""
    run_script: str = ""


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    docker_image: Optional[DockerImage] = Field(default=None, validate_default=True)
    run_script: Optional[File] = Field(default=None, validate_default=True)

    @field_validator("docker_image", mode="before")
    @classmethod
    def set_docker_image_default(cls, value: DockerImage, info: FieldValidationInfo) -> DockerImage:
        if value is None and info.data.get("cmd_args"):
            return DockerImage(url=info.data["cmd_args"].docker_image_url)
        return value

    @field_validator("run_script", mode="before")
    @classmethod
    def set_run_script_default(cls, value: File, info: FieldValidationInfo) -> File:
        if value is None and info.data.get("cmd_args"):
            return File(src=info.data["cmd_args"].run_script)
        return value

    @property
    def installables(self) -> List[Installable]:
        result = [self.run_script]
        if self.docker_image:
            result.append(self.docker_image)
        return result

    @property
    def huggingface_home_host_path(self) -> Path:
        path = Path(self.cmd_args.huggingface_home_host_path)
        if not self.cmd_args.skip_huggingface_home_host_path_validation and not path.is_dir():
            raise FileNotFoundError(f"HuggingFace home path not found at {path}")
        return path
