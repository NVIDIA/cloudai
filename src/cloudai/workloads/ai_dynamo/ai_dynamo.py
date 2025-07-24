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

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cmd: str
    num_nodes: Union[int, list[int]]
    service_args: dict = Field({"workers": 1, "resources": {"gpu": "8"}}, alias="ServiceArgs")
    gpu_memory_utilization: float = Field(0.7, alias="gpu-memory-utilization")
    tensor_parallel_size: Union[int, list[int]] = Field(8, alias="tensor-parallel-size")
    pipeline_parallel_size: Union[int, list[int]] = Field(1, alias="pipeline-parallel-size")
    data_parallel_size: Union[int, list[int]] = Field(1, alias="data-parallel-size")
    enable_expert_parallel: Union[bool, list[bool]] = Field(False, alias="enable-expert-parallel")
    extra_args: str = ""
    enforce_eager: bool = Field(True, alias="enforce-eager")


class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for the VLLM prefill worker."""
    pass


class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for the VLLM decode worker."""
    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="forbid")

    model: str
    workspace_path: Path = Path("/workspace/")
    etcd_cmd: str = "etcd --log-level debug"
    etcd_port: int = 2379
    nats_cmd: str = "nats-server -js"
    nats_port: int = 4222
    ingress_cmd: str
    port: int = 8000
    prefill_worker: PrefillWorkerArgs
    decode_worker: DecodeWorkerArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="forbid")

    port: int = 8000 # Unused
    endpoint: Optional[str] = None
    endpoint_type: str = "kserve"
    streaming: bool
    extra_inputs: Optional[str]
    input_file: Optional[str] = None
    output_tokens_mean: int
    output_tokens_stddev: int = 0
    random_seed: int
    request_count: int
    synthetic_input_tokens_mean: int = 500
    synthetic_input_tokens_stddev: int = 0
    warmup_request_count: int
    concurrency: Optional[Union[int, list[int]]] = None
    request_rate: Optional[Union[float, list[float]]] = None
    iterations: int = 1


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


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    docker_image: Optional[DockerImage] = Field(default=None, validate_default=True)

    @field_validator("docker_image", mode="before")
    @classmethod
    def set_docker_image_default(cls, value: DockerImage, info: FieldValidationInfo) -> DockerImage:
        if value is None and info.data.get("cmd_args"):
            return DockerImage(url=info.data["cmd_args"].docker_image_url)
        return value

    @property
    def installables(self) -> List[Installable]:
        return [self.docker_image] if self.docker_image else []

    @property
    def huggingface_home_host_path(self) -> Path:
        path = Path(self.cmd_args.huggingface_home_host_path)
        if not self.cmd_args.skip_huggingface_home_host_path_validation and not path.is_dir():
            raise FileNotFoundError(f"HuggingFace home path not found at {path}")
        return path
