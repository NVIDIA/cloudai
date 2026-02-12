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
from typing import Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from cloudai.core import (
    DockerImage,
    File,
    GitRepo,
    HFModel,
    Installable,
    JobStatusResult,
    TestRun,
)
from cloudai.models.workload import CmdArgs, TestDefinition


class Args(BaseModel):
    """Arguments for custom workloads."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Workload(BaseModel):
    """Arguments for custom workloads."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    cmd: str
    script: File
    report_name: Optional[str] = Field(default=None, serialization_alias="report-name")
    repo: Optional[GitRepo] = None
    args: Optional[Args] = None
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )

    @model_validator(mode="after")
    def validate_workload(self) -> "Workload":
        """Validate workload."""
        if self.report_name is None:
            self.report_name = f"{self.name}_report.csv"
        if self.args is None:
            self.args = Args()
        return self


class WorkerBaseArgs(Args):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    data_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="data-parallel-size",
        validation_alias=AliasChoices("data-parallel-size", "data_parallel_size"),
    )
    gpu_memory_utilization: float | list[float] | None = Field(
        default=None,
        serialization_alias="gpu-memory-utilization",
        validation_alias=AliasChoices("gpu-memory-utilization", "gpu_memory_utilization"),
    )
    pipeline_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="pipeline-parallel-size",
        validation_alias=AliasChoices("pipeline-parallel-size", "pipeline_parallel_size"),
    )
    tensor_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="tensor-parallel-size",
        validation_alias=AliasChoices("tensor-parallel-size", "tensor_parallel_size"),
    )


class WorkerConfig(BaseModel):
    """Configuration for workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

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


class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for prefill worker."""

    pass


class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for decode worker."""

    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = "Qwen/Qwen3-0.6B"
    backend: str = "vllm"
    connector: Optional[str] = None  # none, lmcache, kvbm
    workspace_path: str = Field(
        default="/workspace",
        serialization_alias="workspace-path",
        validation_alias=AliasChoices("workspace-path", "workspace_path"),
    )
    port: int = Field(
        default=8000,
        description="Dynamo frontend HTTP API port",
    )
    etcd_port: int = Field(
        default=2379,
        serialization_alias="etcd-port",
        validation_alias=AliasChoices("etcd-port", "etcd_port"),
    )
    nats_port: int = Field(
        default=4222,
        serialization_alias="nats-port",
        validation_alias=AliasChoices("nats-port", "nats_port"),
    )
    decode_worker: WorkerConfig = Field(default_factory=WorkerConfig)
    decode_cmd: str = Field(
        default="python3 -m dynamo.vllm",
        serialization_alias="decode-cmd",
        validation_alias=AliasChoices("decode-cmd", "decode_cmd"),
    )
    prefill_worker: WorkerConfig = Field(default_factory=WorkerConfig)
    prefill_cmd: str = Field(
        default="python3 -m dynamo.vllm --is-prefill-worker",
        serialization_alias="prefill-cmd",
        validation_alias=AliasChoices("prefill-cmd", "prefill_cmd"),
    )


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
    extra_config_nixl_path: str = "%CACHEDIR%"

    # LMCache controller configuration
    enable_controller: bool = True
    lmcache_instance_id: str = "lmcache_default_instance"
    controller_url: str = "localhost:9001"
    lmcache_worker_port: int = 8788
    distributed_url: str = "localhost:8789"


class LMCache(BaseModel):
    """LMCache configuration."""

    model_config = ConfigDict(extra="allow")

    controller_cmd: str = "lmcache_controller --host localhost --port 9000 --monitor-port 9001"
    repo: Optional[GitRepo] = GitRepo(
        url="git@github.com:LMCache/LMCache.git",
        commit="ab8530993992db873869ba882320953582d94309",
        mount_as="/git/LMCache",
    )

    args: LMCacheArgs = Field(default_factory=LMCacheArgs)
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class GenAIPerf(Workload):
    """Workload configuration for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    name: str = "genai_perf"
    cmd: str = "genai-perf profile"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")


class AIPerfArgs(Args):
    """Arguments for AIPerf profiling - alternative to GenAI-Perf."""

    concurrency: int | None = Field(default=None)
    request_rate: float | None = Field(
        default=None,
        serialization_alias="request-rate",
        validation_alias=AliasChoices("request-rate", "request_rate"),
    )
    synthetic_input_tokens_mean: int | None = Field(
        default=None,
        serialization_alias="synthetic-input-tokens-mean",
        validation_alias=AliasChoices("synthetic-input-tokens-mean", "synthetic_input_tokens_mean"),
    )
    synthetic_input_tokens_stddev: int = Field(
        default=0,
        serialization_alias="synthetic-input-tokens-stddev",
        validation_alias=AliasChoices("synthetic-input-tokens-stddev", "synthetic_input_tokens_stddev"),
    )
    output_tokens_mean: int | None = Field(
        default=None,
        serialization_alias="output-tokens-mean",
        validation_alias=AliasChoices("output-tokens-mean", "output_tokens_mean"),
    )
    output_tokens_stddev: int = Field(
        default=0,
        serialization_alias="output-tokens-stddev",
        validation_alias=AliasChoices("output-tokens-stddev", "output_tokens_stddev"),
    )
    request_count: int | None = Field(
        default=None,
        serialization_alias="request-count",
        validation_alias=AliasChoices("request-count", "request_count"),
    )
    benchmark_duration: int | None = Field(
        default=None,
        serialization_alias="benchmark-duration",
        validation_alias=AliasChoices("benchmark-duration", "benchmark_duration"),
    )
    streaming: bool = Field(default=True)
    warmup_request_count: int = Field(
        default=10,
        serialization_alias="warmup-request-count",
        validation_alias=AliasChoices("warmup-request-count", "warmup_request_count"),
    )
    endpoint_type: str = Field(
        default="chat",
        serialization_alias="endpoint-type",
        validation_alias=AliasChoices("endpoint-type", "endpoint_type"),
    )
    ui_type: str = Field(
        default="simple",
        serialization_alias="ui-type",
        validation_alias=AliasChoices("ui-type", "ui_type"),
    )
    export_level: Literal["summary", "records", "raw"] = Field(
        default="records",
        serialization_alias="export-level",
        validation_alias=AliasChoices("export-level", "export_level"),
        description=(
            "Controls output detail: summary (aggregate only),"
            " records (per-request metrics), raw (full request/response data)"
        ),
    )
    slice_duration: float | None = Field(
        default=5.0,
        serialization_alias="slice-duration",
        validation_alias=AliasChoices("slice-duration", "slice_duration"),
        description="Duration in seconds for time-sliced metric analysis. Enables bar chart visualizations.",
    )

    # Multi-turn / Agentic mode parameters
    conversation_num: int | None = Field(
        default=None,
        serialization_alias="conversation-num",
        validation_alias=AliasChoices("conversation-num", "conversation_num"),
        description="Total number of conversation sessions for multi-turn benchmarks.",
    )
    conversation_turn_mean: int | None = Field(
        default=None,
        serialization_alias="conversation-turn-mean",
        validation_alias=AliasChoices("conversation-turn-mean", "conversation_turn_mean"),
        description="Average number of turns (steps) per conversation session.",
    )
    conversation_turn_stddev: int | None = Field(
        default=None,
        serialization_alias="conversation-turn-stddev",
        validation_alias=AliasChoices("conversation-turn-stddev", "conversation_turn_stddev"),
        description="Standard deviation for turn counts per session.",
    )
    conversation_turn_delay_mean: int | None = Field(
        default=None,
        serialization_alias="conversation-turn-delay-mean",
        validation_alias=AliasChoices("conversation-turn-delay-mean", "conversation_turn_delay_mean"),
        description="Mean delay between turns in milliseconds (simulates user think time).",
    )
    conversation_turn_delay_stddev: int | None = Field(
        default=None,
        serialization_alias="conversation-turn-delay-stddev",
        validation_alias=AliasChoices("conversation-turn-delay-stddev", "conversation_turn_delay_stddev"),
        description="Standard deviation for turn delays in milliseconds.",
    )
    turn_sequence: str | None = Field(
        default=None,
        serialization_alias="turn-sequence",
        validation_alias=AliasChoices("turn-sequence", "turn_sequence"),
        description=(
            "Explicit ISL/OSL pairs for each turn. Format: 'ISL,OSL;ISL,OSL;...' "
            "Example: '1024,100;2048,100;3072,200' means turn 1=ISL 1024/OSL 100, "
            "turn 2=ISL 2048/OSL 100, etc. Cycles if more turns than defined pairs."
        ),
    )


class AIPerf(Workload):
    """Workload configuration for AIPerf."""

    model_config = ConfigDict(extra="allow")

    name: str = "aiperf"
    cmd: str = "aiperf profile"
    args: Optional[Args] = Field(default_factory=AIPerfArgs)
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/aiperf.sh")

    extra_args: str | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class LMBench(Workload):
    """Workload configuration for LMBench."""

    model_config = ConfigDict(extra="allow")

    name: str = "lmbench"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/lmbench.sh")
    cmd: str = "python3 ./synthetic-multi-round-qa/multi-round-qa.py"
    qps: str | list[str] | None = "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0"
    repo: Optional[GitRepo] = GitRepo(
        url="git@github.com:LMCache/LMBenchmark.git",
        commit="e1406623c5e88878cf2b7fbd64fe6c47f7dcb66f",
        mount_as="/git/LMBenchmark",
    )


class KVStorage(Workload):
    """KV storage workload script."""

    model_config = ConfigDict(extra="allow")

    name: str = "kvstorage"
    cmd: str = "hostname"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/kvstorage.sh")


class Constraints(BaseModel):
    """Constraints for validation of AI Dynamo configurations when using DSE."""

    model_config = ConfigDict(extra="allow")

    prefill_tp_le_decode_tp: bool = True
    tp_times_pp_le_gpus_per_node: bool = True
    prefill_decode_nodes_le_total_nodes: bool = True


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    hf_home_path: Optional[str] = Field(default=None, serialization_alias="hf_home_path")
    storage_cache_dir: Optional[str] = Field(default=None, serialization_alias="storage_cache_dir")
    num_nodes: int = 1
    gpus_per_node: int = 8
    dynamo: AIDynamoArgs
    lmcache: LMCache = Field(default_factory=LMCache)
    genai_perf: GenAIPerf = Field(default_factory=GenAIPerf)
    aiperf: AIPerf = Field(default_factory=AIPerf)
    lmbench: LMBench = Field(default_factory=LMBench)
    kvstorage: KVStorage = Field(default_factory=KVStorage)
    workloads: str = "genai_perf.sh,aiperf.sh,lmbench.sh,kvstorage.sh"


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    genai_perf_script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")
    aiperf_script: File = File(Path(__file__).parent.parent / "ai_dynamo/aiperf.sh")
    calc_percentile_csv: File = File(Path(__file__).parent.parent / "ai_dynamo/calc_percentile_csv.py")
    dynamo_repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git",
        commit="f7e468c7e8ff0d1426db987564e60572167e8464",
        mount_as="/git/dynamo",
    )
    _hf_model: HFModel | None = None
    constraints: Constraints = Constraints()

    def success_marker(self) -> str:
        return "success-marker.txt"

    def failure_marker(self) -> str:
        return "failure-marker.txt"

    def get_workload_map(self) -> dict[str, Workload]:
        """Get a map of workload scripts to workload objects."""
        return {
            self.cmd_args.genai_perf.script.src.name: self.cmd_args.genai_perf,
            self.cmd_args.aiperf.script.src.name: self.cmd_args.aiperf,
            self.cmd_args.lmbench.script.src.name: self.cmd_args.lmbench,
            self.cmd_args.kvstorage.script.src.name: self.cmd_args.kvstorage,
        }

    @model_validator(mode="after")
    def validate_test_definition(self) -> "AIDynamoTestDefinition":
        """Validate test definition."""
        # Populate git_repos list with all git repositories used by this test definition.
        self.git_repos = [self.dynamo_repo]
        if self.cmd_args.lmcache.repo:
            self.git_repos.append(self.cmd_args.lmcache.repo)
        if self.cmd_args.lmbench.repo:
            self.git_repos.append(self.cmd_args.lmbench.repo)
        if self.cmd_args.kvstorage.repo:
            self.git_repos.append(self.cmd_args.kvstorage.repo)

        # Validate benchmark names
        workloads = self.cmd_args.workloads.split(",")
        for workload in workloads:
            if workload not in [
                self.cmd_args.genai_perf.script.src.name,
                self.cmd_args.aiperf.script.src.name,
                self.cmd_args.lmbench.script.src.name,
                self.cmd_args.kvstorage.script.src.name,
            ]:
                raise ValueError(f"Invalid workload script: {workload}")

        return self

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
        result = [
            self.docker_image,
            self.script,
            # self.hf_model,
            self.genai_perf_script,
            self.aiperf_script,
            self.calc_percentile_csv,
            self.cmd_args.lmbench.script,
            self.cmd_args.kvstorage.script,
            File(Path(__file__).parent.parent / "ai_dynamo/kvstorage.py"),
            *self.git_repos,
        ]

        return result

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        result = True
        workload_map = self.get_workload_map()
        failure_marker = output_path / self.failure_marker()
        success_marker = output_path / self.success_marker()

        if failure_marker.exists():
            return JobStatusResult(False, error_message=f"Failure marker file found with contents: \n{failure_marker.read_text()}")

        if not success_marker.exists():
            return JobStatusResult(False, error_message=f"Success marker file not found: {success_marker.absolute()}")

        for workload in self.cmd_args.workloads.split(","):
            if not workload_map.get(workload):
                logging.info(f"Workload {workload} not found in workload map")
                result = False
                continue
            report_name = workload_map[workload].report_name
            assert report_name is not None
            workload_csv_file = output_path / report_name
            if not workload_csv_file.exists():
                logging.info(f"Result file ({workload_csv_file.absolute()}) not found for workload: {workload}")
                result = False
            else:
                logging.info(f"Result file ({workload_csv_file.absolute()}) exists for {workload}")

        return JobStatusResult(result)

    def constraint_check(self, tr: TestRun) -> bool:
        prefill_worker = tr.test.cmd_args.dynamo.prefill_worker
        decode_worker = tr.test.cmd_args.dynamo.decode_worker

        prefill_tp = prefill_worker.args.tensor_parallel_size if prefill_worker else 1
        decode_tp = decode_worker.args.tensor_parallel_size if decode_worker else 1
        prefill_pp = prefill_worker.args.pipeline_parallel_size if prefill_worker else 1
        decode_pp = decode_worker.args.pipeline_parallel_size if decode_worker else 1
        prefill_nodes = prefill_worker.num_nodes if prefill_worker else 0
        decode_nodes = decode_worker.num_nodes if decode_worker else 1

        if self.constraints.prefill_tp_le_decode_tp and prefill_tp > decode_tp:
            logging.info("constraint_check failed for: prefill_tp_le_decode_tp")
            return False
        logging.info("constraint_check passed for: prefill_tp_le_decode_tp")

        gpus_per_node = tr.test.cmd_args.gpus_per_node
        if self.constraints.tp_times_pp_le_gpus_per_node and (
            prefill_tp * prefill_pp > gpus_per_node or decode_tp * decode_pp > gpus_per_node
        ):
            logging.info("constraint_check failed for: tp_times_pp_le_gpus_per_node")
            return False
        logging.info("constraint_check passed for: tp_times_pp_le_gpus_per_node")

        num_nodes = tr.test.cmd_args.num_nodes
        nodes_check = self.constraints.prefill_decode_nodes_le_total_nodes
        if nodes_check and prefill_nodes + decode_nodes > num_nodes:
            logging.info("constraint_check failed for: prefill_decode_nodes_le_total_nodes")
            return False
        logging.info("constraint_check passed for: prefill_decode_nodes_le_total_nodes")

        return True
