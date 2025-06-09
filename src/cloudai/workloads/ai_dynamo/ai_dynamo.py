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

from pydantic import BaseModel, Field

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class FrontendArgs(BaseModel):
    """Arguments for the frontend node."""

    endpoint: str = "dynamo.Processor.chat/completions"
    port: int = 8000
    port_etcd: int = Field(2379, alias="port_etcd")
    port_nats: int = Field(4222, alias="port_nats")


class ProcessorArgs(BaseModel):
    """Arguments for the processor node."""

    block_size: int = Field(64, alias="block-size")
    max_model_len: int = Field(8192, alias="max-model-len")
    router: str = "kv"


class RouterArgs(BaseModel):
    """Arguments for the router."""

    min_workers: int = Field(1, alias="min-workers")


class PrefillWorkerArgs(BaseModel):
    """Arguments for the prefill worker node."""

    num_nodes: Union[int, list[int]]
    kv_transfer_config: str = Field('{"kv_connector":"DynamoNixlConnector"}', alias="kv-transfer-config")
    block_size: int = Field(64, alias="block-size")
    max_model_len: int = Field(8192, alias="max-model-len")
    max_num_seqs: int = Field(16, alias="max-num-seqs")
    gpu_memory_utilization: float = Field(0.95, alias="gpu-memory-utilization")
    tensor_parallel_size: Union[int, list[int]] = Field(8, alias="tensor-parallel-size")
    quantization: str = "modelopt"
    service_args: dict = Field({"workers": 1, "resources": {"gpu": "8"}}, alias="ServiceArgs")


class VllmWorkerArgs(BaseModel):
    """Arguments for the VllmWorker node."""

    num_nodes: Union[int, list[int]]
    kv_transfer_config: str = Field('{"kv_connector":"DynamoNixlConnector"}', alias="kv-transfer-config")
    block_size: int = Field(64, alias="block-size")
    max_model_len: int = Field(8192, alias="max-model-len")
    max_num_seqs: int = Field(16, alias="max-num-seqs")
    remote_prefill: bool = Field(True, alias="remote-prefill")
    conditional_disagg: bool = Field(True, alias="conditional-disagg")
    max_local_prefill_length: int = Field(10, alias="max-local-prefill-length")
    max_prefill_queue_size: int = Field(2, alias="max-prefill-queue-size")
    gpu_memory_utilization: float = Field(0.95, alias="gpu-memory-utilization")
    tensor_parallel_size: Union[int, list[int]] = Field(8, alias="tensor-parallel-size")
    router: str = "kv"
    quantization: str = "modelopt"
    enable_prefix_caching: bool = Field(True, alias="enable-prefix-caching")
    service_args: dict = Field({"workers": 1, "resources": {"gpu": "8"}}, alias="ServiceArgs")


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    frontend: FrontendArgs
    processor: ProcessorArgs
    router: RouterArgs
    prefill_worker: PrefillWorkerArgs
    vllm_worker: VllmWorkerArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

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
