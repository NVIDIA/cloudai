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
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class CommonConfig(BaseModel):
    """Common configuration shared across components."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str
    kv_transfer_config: str = Field('{"kv_connector":"NixlConnector","kv_role":"kv_both"}', alias="kv-transfer-config")
    served_model_name: str


class FrontendArgs(BaseModel):
    """Arguments for the frontend node."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    endpoint: str = "dynamo.SimpleLoadBalancer.generate_disagg"
    port: int = 8000
    port_etcd: int = 2379
    port_nats: int = 4222


class SimpleLoadBalancerArgs(BaseModel):
    """Arguments for the load balancer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enable_disagg: bool = True


class VllmWorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_nodes: Union[int, list[int]]
    service_args: dict = Field({"workers": 1, "resources": {"gpu": "8"}}, alias="ServiceArgs")
    gpu_memory_utilization: float = Field(0.7, alias="gpu-memory-utilization")
    tensor_parallel_size: int = Field(8, alias="tensor-parallel-size")
    pipeline_parallel_size: int = Field(1, alias="pipeline-parallel-size")
    enforce_eager: bool = Field(True, alias="enforce-eager")


class VllmPrefillWorkerArgs(VllmWorkerBaseArgs):
    """Arguments for the VLLM prefill worker."""

    pass


class VllmDecodeWorkerArgs(VllmWorkerBaseArgs):
    """Arguments for the VLLM decode worker."""

    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="forbid")

    common: CommonConfig
    frontend: FrontendArgs = FrontendArgs()
    simple_load_balancer: SimpleLoadBalancerArgs = SimpleLoadBalancerArgs()
    vllm_prefill_worker: VllmPrefillWorkerArgs
    vllm_decode_worker: VllmDecodeWorkerArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="forbid")

    port: int = 8000
    endpoint: Optional[str] = None
    endpoint_type: str = "kserve"
    streaming: bool
    extra_inputs: Optional[str]
    input_file: Optional[str] = None
    output_tokens_mean: int
    osl: int = -1
    output_tokens_stddev: int = 0
    random_seed: int
    request_count: int
    synthetic_input_tokens_mean: int
    isl: int = 550
    synthetic_input_tokens_stddev: int = 0
    warmup_request_count: int
    concurrency: Optional[Union[int, list[int]]] = None
    request_rate: Optional[Union[float, list[float]]] = None


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    served_model_name: str
    dynamo: AIDynamoArgs
    sleep_seconds: int = 660
    genai_perf: GenAIPerfArgs


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> List[Installable]:
        return [self.docker_image]

    @property
    def hugging_face_home_path(self) -> Path:
        raw = self.extra_env_vars.get("HF_HOME")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("HF_HOME must be set and non-empty")
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"HF_HOME path not found at {path}")
        return path
