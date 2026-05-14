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

import logging
from pathlib import Path
from typing import Literal, Optional, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from cloudai.core import (
    DockerImage,
    File,
    GitRepo,
    HFModel,
    Installable,
    JobStatusResult,
    System,
    TestRun,
)
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmSystem


class Args(BaseModel):
    """Arguments for custom workloads."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Workload(BaseModel):
    """Workload configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    cmd: str
    script: File
    report_name: str = Field(
        default_factory=lambda self: f"{self['name']}_report.csv",
        serialization_alias="report-name",
        validation_alias=AliasChoices("report-name", "report_name"),
    )
    repo: Optional[GitRepo] = None
    args: Args = Field(default_factory=Args)
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class WorkerBaseArgs(Args):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Used by VLLM backend.
    model: str | None = None

    # Used by SGLang/SGLang-DSR1 backends.
    model_path: str | None = Field(default=None, serialization_alias="model-path")
    served_model_name: str | None = Field(default=None, serialization_alias="served-model-name")

    gpu_memory_utilization: float | list[float] | None = Field(
        default=None,
        serialization_alias="gpu-memory-utilization",
        validation_alias=AliasChoices("gpu-memory-utilization", "gpu_memory_utilization"),
    )
    pipeline_parallel_size: int | list[int] = Field(
        default=1,
        serialization_alias="pipeline-parallel-size",
        validation_alias=AliasChoices("pipeline-parallel-size", "pipeline_parallel_size"),
    )
    tensor_parallel_size: int | list[int] = Field(
        default=1,
        serialization_alias="tensor-parallel-size",
        validation_alias=AliasChoices("tensor-parallel-size", "tensor_parallel_size"),
    )
    data_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="data-parallel-size",
        validation_alias=AliasChoices("data-parallel-size", "data_parallel_size"),
    )


class WorkerConfig(BaseModel):
    """Configuration for workers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cmd: str
    worker_initialized_regex: str = Field(
        validation_alias=AliasChoices("worker-initialized-regex", "worker_initialized_regex"),
        serialization_alias="worker-initialized-regex",
    )
    multiple_workers_per_node: bool = Field(
        default=False,
        validation_alias=AliasChoices("multiple-workers-per-node", "multiple_workers_per_node"),
        serialization_alias="multiple-workers-per-node",
    )

    num_nodes: int | list[int] = Field(
        default=1, serialization_alias="num-nodes", validation_alias=AliasChoices("num-nodes", "num_nodes")
    )
    nodes: str | None = Field(default=None)

    args: WorkerBaseArgs = Field(default_factory=WorkerBaseArgs)

    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str = "Qwen/Qwen3-0.6B"
    backend: Literal["vllm", "sglang", "sglang_dsr1"] = "vllm"
    endpoint: str = Field(default="v1/chat/completions")
    connector: Optional[str | list[str]] = None

    @field_validator("connector", mode="before")
    @classmethod
    def validate_connector(cls, v: str | list[str] | None) -> str | list[str] | None:
        if v is None:
            return v
        allowed_connectors = ["kvbm", "lmcache", "nixl", "none"]

        # Connectors can be either a single string or a space-separated list.
        values = v if isinstance(v, str) else " ".join(v)
        values = [c.strip() for c in values.split(" ")]

        for connector in values:
            if connector not in allowed_connectors:
                raise ValueError(f"Invalid connector: {connector}. Available connectors: {allowed_connectors}")
        return v

    workspace_path: str = Field(
        default="/workspace",
        serialization_alias="workspace-path",
        validation_alias=AliasChoices("workspace-path", "workspace_path"),
    )
    ingress_cmd: str = Field(
        default="python -m dynamo.frontend --router-mode kv",
        serialization_alias="ingress-cmd",
        validation_alias=AliasChoices("ingress-cmd", "ingress_cmd"),
    )
    node_setup_cmd: str = Field(
        default="/usr/local/ucx/bin/ucx_info -d |grep Transport | sort -u;",
        serialization_alias="node-setup-cmd",
        validation_alias=AliasChoices("node-setup-cmd", "node_setup_cmd"),
    )
    port: int = Field(
        default=8000,
        description="Dynamo frontend HTTP API port",
    )
    etcd_cmd: str = Field(
        default="etcd --log-level info --data-dir /tmp/etcd",
        serialization_alias="etcd-cmd",
        validation_alias=AliasChoices("etcd-cmd", "etcd_cmd"),
    )
    etcd_port: int = Field(
        default=2379,
        serialization_alias="etcd-port",
        validation_alias=AliasChoices("etcd-port", "etcd_port"),
    )
    nats_cmd: str = Field(
        default="nats-server -js",
        serialization_alias="nats-cmd",
        validation_alias=AliasChoices("nats-cmd", "nats_cmd"),
    )
    nats_port: int = Field(
        default=4222,
        serialization_alias="nats-port",
        validation_alias=AliasChoices("nats-port", "nats_port"),
    )

    decode_worker: WorkerConfig = WorkerConfig(
        cmd="python3 -m dynamo.vllm",
        worker_initialized_regex="VllmWorker.*has.been.initialized",
    )
    prefill_worker: WorkerConfig = WorkerConfig(
        cmd="python3 -m dynamo.vllm --is-prefill-worker",
        worker_initialized_regex="VllmWorker.*has.been.initialized",
    )

    @model_validator(mode="after")
    def populate_prefill_decode_args(self) -> "AIDynamoArgs":
        """Populate prefill/decode args."""
        if self.backend.lower() == "vllm":
            self.prefill_worker.args.model = self.model
            self.decode_worker.args.model = self.model
        elif self.backend.lower() in ["sglang", "sglang_dsr1"]:
            self.prefill_worker.args.model_path = self.model
            self.decode_worker.args.model_path = self.model
            self.prefill_worker.args.served_model_name = self.model
            self.decode_worker.args.served_model_name = self.model
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        return self


class LMCacheArgs(BaseModel):
    """Arguments for LMCache."""

    model_config = ConfigDict(extra="allow")

    chunk_size: int = 256
    local_cpu: bool = False
    nixl_buffer_size: int = 10737418240
    nixl_buffer_device: str = "cuda"
    extra_config_enable_nixl_storage: bool = True
    extra_config_nixl_backend: str = "GDS_MT"
    extra_config_nixl_file_pool_size: int = 64

    # LMCache controller configuration
    enable_controller: bool = True
    lmcache_instance_id: str = "lmcache_default_instance"
    controller_url: str = "localhost:9001"
    lmcache_worker_port: int = 8788
    distributed_url: str = "localhost:8789"


class LMCache(BaseModel):
    """LMCache configuration."""

    model_config = ConfigDict(extra="forbid")

    controller_cmd: str = "lmcache_controller --host localhost --port 9000 --monitor-port 9001"
    repo: GitRepo = GitRepo(
        url="https://github.com/LMCache/LMCache.git", commit="ab8530993992db873869ba882320953582d94309"
    )

    args: LMCacheArgs = Field(default_factory=LMCacheArgs)
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )

    @property
    def installables(self) -> list[Installable]:
        return [self.repo]


class GenAIPerf(Workload):
    """Workload configuration for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    name: str = "genai_perf"
    cmd: str = "genai-perf profile"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")
    client_docker_image_url: str | None = Field(
        default=None,
        serialization_alias="client-docker-image-url",
        validation_alias=AliasChoices("client-docker-image-url", "client_docker_image_url"),
    )

    @property
    def installables(self) -> list[Installable]:
        result: list[Installable] = [self.script]
        if self.client_docker_image_url:
            result.append(DockerImage(url=self.client_docker_image_url))
        return result


class Constraints(BaseModel):
    """Constraints for validation of AI Dynamo configurations when using DSE."""

    model_config = ConfigDict(extra="forbid")

    prefill_tp_le_decode_tp: bool = True
    tp_times_pp_le_gpus_per_node: bool = True


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    model_config = ConfigDict(extra="forbid")

    docker_image_url: str
    storage_cache_dir: Optional[str | list[str]] = Field(default="/tmp", serialization_alias="storage_cache_dir")
    dynamo: AIDynamoArgs
    lmcache: LMCache = Field(default_factory=LMCache)
    genai_perf: GenAIPerf = Field(default_factory=GenAIPerf)
    workloads: str = "genai_perf.sh"

    @field_validator("workloads", mode="before")
    @classmethod
    def validate_workloads(cls, v: str) -> str:
        allowed_workloads = ["genai_perf.sh"]
        values = [w.strip() for w in v.split(",")]
        for workload in values:
            if workload not in allowed_workloads:
                raise ValueError(f"Invalid workload: {workload}. Available workloads: {allowed_workloads}")
        return ",".join(values)

    @property
    def workloads_list(self) -> list[str]:
        return [w.strip() for w in self.workloads.split(",")]

    @property
    def installables(self) -> list[Installable]:
        return [
            *self.lmcache.installables,
            *self.genai_perf.installables,
        ]


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    model_config = ConfigDict(extra="forbid")

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git", commit="f7e468c7e8ff0d1426db987564e60572167e8464"
    )
    _hf_model: HFModel | None = None
    constraints: Constraints = Constraints()

    success_marker: str = "success-marker.txt"
    failure_marker: str = "failure-marker.txt"

    @model_validator(mode="after")
    def workload_scripts(self) -> "AIDynamoTestDefinition":
        """Populate prefill/decode args."""
        workload_map = self.get_workload_map()
        for workload in self.cmd_args.workloads_list:
            if workload not in workload_map:
                raise ValueError(f"Invalid workload: {workload}. Available workloads: {list(workload_map.keys())}")

        return self

    def get_workload_map(self) -> dict[str, Workload]:
        """Get a map of workload scripts to workload objects."""
        return {
            self.cmd_args.genai_perf.script.src.name: self.cmd_args.genai_perf,
        }

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            logging.info(f"Creating HFModel for: {self.cmd_args.dynamo.model}")
            self._hf_model = HFModel(model_name=self.cmd_args.dynamo.model)
        return self._hf_model

    @property
    def installables(self) -> list[Installable]:
        """Get all installables for this test definition."""
        return [
            self.docker_image,
            self.repo,
            self.script,
            self.hf_model,
            *self.cmd_args.installables,
        ]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        result = True
        workload_map = self.get_workload_map()
        failure_marker = output_path / self.failure_marker
        success_marker = output_path / self.success_marker

        if failure_marker.exists():
            contents = failure_marker.read_text()
            return JobStatusResult(False, error_message=f"Failure marker found:\n{contents}")

        if not success_marker.exists():
            return JobStatusResult(False, error_message=f"Success marker file not found: {success_marker.absolute()}")

        for workload in self.cmd_args.workloads_list:
            if workload not in workload_map:
                logging.info(f"Workload {workload} not found in workload map")
                result = False
                continue
            report_name = workload_map[workload].report_name
            if report_name is None:
                logging.warning(f"Workload {workload} has no report_name configured")
                result = False
                continue
            workload_csv_file = output_path / report_name
            if not workload_csv_file.exists():
                logging.info(f"Result file ({workload_csv_file.absolute()}) not found for workload: {workload}")
                result = False
            else:
                logging.info(f"Result file ({workload_csv_file.absolute()}) exists for {workload}")

        return JobStatusResult(result)

    def constraint_check(self, tr: TestRun, system: Optional[System]) -> bool:
        prefill_worker = tr.test.cmd_args.dynamo.prefill_worker
        decode_worker = tr.test.cmd_args.dynamo.decode_worker

        prefill_tp = prefill_worker.args.tensor_parallel_size
        prefill_pp = prefill_worker.args.pipeline_parallel_size

        decode_tp = decode_worker.args.tensor_parallel_size
        decode_pp = decode_worker.args.pipeline_parallel_size

        if self.constraints.prefill_tp_le_decode_tp and prefill_tp > decode_tp:
            logging.info("constraint_check failed for: prefill_tp_le_decode_tp")
            return False
        logging.info("constraint_check passed for: prefill_tp_le_decode_tp")

        gpus_per_node = 0
        slurm_system = cast(SlurmSystem, system)
        if slurm_system and slurm_system.gpus_per_node:
            gpus_per_node = slurm_system.gpus_per_node

        if (
            gpus_per_node > 0
            and self.constraints.tp_times_pp_le_gpus_per_node
            and (prefill_tp * prefill_pp > gpus_per_node or decode_tp * decode_pp > gpus_per_node)
        ):
            logging.info("constraint_check failed for: tp_times_pp_le_gpus_per_node")
            return False
        logging.info("constraint_check passed for: tp_times_pp_le_gpus_per_node")

        return True
