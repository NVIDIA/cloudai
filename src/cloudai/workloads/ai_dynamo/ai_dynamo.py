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
from typing import Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
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

from .report_generation_strategy import CSV_FILES_PATTERN, JSON_FILES_PATTERN


class BenchmarkArgs(BaseModel):
    """Arguments for custom benchmarks."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Benchmark(BaseModel):
    """Arguments for custom benchmarks."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    cmd: str
    script: File
    report_name: Optional[str] = Field(default=None, serialization_alias="report-name")
    repo: Optional[GitRepo] = None
    enabled: bool = False
    args: Optional[BenchmarkArgs] = None
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )

    @model_validator(mode="after")
    def set_default_report_name(self) -> "Benchmark":
        """Set default report_name based on name if not provided."""
        if self.report_name is None:
            self.report_name = f"{self.name}_report.csv"
        return self

    @field_serializer("repo")
    def _serialize_repo(self, v: GitRepo) -> str:
        return v.container_mount if v else "/invalid/repo/path"

    @field_serializer("script")
    def _serialize_script(self, v: File) -> str:
        return v.src.name


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: int | list[int] = Field(
        default=1, serialization_alias="num-nodes", validation_alias=AliasChoices("num-nodes", "num_nodes")
    )
    nodes: str | None = Field(default=None)

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
    decode_worker: DecodeWorkerArgs = Field(default_factory=DecodeWorkerArgs)
    decode_cmd: str = Field(
        default="python3 -m dynamo.vllm",
        serialization_alias="decode-cmd",
        validation_alias=AliasChoices("decode-cmd", "decode_cmd"),
    )
    prefill_worker: PrefillWorkerArgs | None = None
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

    @field_serializer("repo")
    def _serialize_repo(self, v: GitRepo) -> str:
        return v.container_mount if v else "/invalid/repo/path"


class GenAIPerf(Benchmark):
    """Benchmark configuration for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    name: str = "genai_perf"
    cmd: str = "genai-perf profile"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")
    enabled: bool = True


class LMBench(Benchmark):
    """Benchmark configuration for LMBench."""

    model_config = ConfigDict(extra="allow")

    name: str = "lmbench"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/lmbench.sh")
    cmd: str = "python3 ./synthetic-multi-round-qa/multi-round-qa.py"
    repo: Optional[GitRepo] = GitRepo(
        url="git@github.com:LMCache/LMBenchmark.git",
        commit="e1406623c5e88878cf2b7fbd64fe6c47f7dcb66f",
        mount_as="/git/LMBenchmark",
    )


class CustomBench(Benchmark):
    """Generic benchmark script."""

    model_config = ConfigDict(extra="allow")

    name: str = "custom_bench"
    cmd: str = "hostname"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/custom_bench.sh")
    enabled: bool = False


class Constraints(BaseModel):
    """Constraints for validation of AI Dynamo configurations when using DSE."""

    model_config = ConfigDict(extra="allow")

    prefill_tp_le_decode_tp: bool = True
    tp_times_pp_le_gpus_per_node: bool = True
    prefill_decode_nodes_le_total_nodes: bool = True


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    storage_cache_dir: Optional[str] = None
    num_nodes: int = 1
    gpus_per_node: int = 8
    dynamo: AIDynamoArgs
    lmcache: LMCache = Field(default_factory=LMCache)
    genai_perf: GenAIPerf = Field(default_factory=GenAIPerf)
    lmbench: LMBench = Field(default_factory=LMBench)
    custom_bench: CustomBench = Field(default_factory=CustomBench)


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    genai_perf_script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")
    calc_percentile_csv: File = File(Path(__file__).parent.parent / "ai_dynamo/calc_percentile_csv.py")
    dynamo_repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git",
        commit="f7e468c7e8ff0d1426db987564e60572167e8464",
        mount_as="/git/dynamo",
    )
    _hf_model: HFModel | None = None
    benchmarks: str = "genai_perf"
    constraints: Constraints = Constraints()

    @model_validator(mode="after")
    def populate_git_repos(self) -> "AIDynamoTestDefinition":
        """Populate git_repos list with all git repositories used by this test definition."""
        self.git_repos = [self.dynamo_repo]
        if self.cmd_args.lmcache.repo:
            self.git_repos.append(self.cmd_args.lmcache.repo)
        if self.cmd_args.lmbench.repo:
            self.git_repos.append(self.cmd_args.lmbench.repo)
        if self.cmd_args.custom_bench.repo:
            self.git_repos.append(self.cmd_args.custom_bench.repo)
        return self

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            self._hf_model = HFModel(model_name=self.cmd_args.dynamo.model)
        return self._hf_model

    @property
    def installables(self) -> list[Installable]:
        """Get all installables for this test definition."""
        result = [
            self.docker_image,
            self.script,
            self.hf_model,
            self.genai_perf_script,
            self.calc_percentile_csv,
            self.cmd_args.lmbench.script,
            self.cmd_args.custom_bench.script,
            File(Path(__file__).parent.parent / "ai_dynamo/openai_chat_client.py"),
            *self.git_repos,
        ]

        return result

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        csv_files = list(output_path.rglob(CSV_FILES_PATTERN))
        json_files = list(output_path.rglob(JSON_FILES_PATTERN))
        logging.debug(f"Found CSV files in {output_path.absolute()}: {csv_files}, JSON files: {json_files}")
        has_results = len(csv_files) > 0 and len(json_files) > 0
        if not has_results:
            return JobStatusResult(False, "No result files found in the output directory.")
        return JobStatusResult(True)

    def constraint_check(self, tr: TestRun) -> bool:
        prefill_worker = tr.test.cmd_args.dynamo.prefill_worker
        decode_worker = tr.test.cmd_args.dynamo.decode_worker

        prefill_tp = prefill_worker.tensor_parallel_size if prefill_worker else 1
        decode_tp = decode_worker.tensor_parallel_size if decode_worker else 1
        prefill_pp = prefill_worker.pipeline_parallel_size if prefill_worker else 1
        prefill_nodes = prefill_worker.num_nodes if prefill_worker else 0
        decode_nodes = decode_worker.num_nodes if decode_worker else 1

        if self.constraints.prefill_tp_le_decode_tp and prefill_tp > decode_tp:
            logging.info("constraint_check failed for: prefill_tp_le_decode_tp")
            return False
        logging.info("constraint_check passed for: prefill_tp_le_decode_tp")

        gpus_per_node = tr.test.cmd_args.gpus_per_node
        if self.constraints.tp_times_pp_le_gpus_per_node and prefill_tp * prefill_pp > gpus_per_node:
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
