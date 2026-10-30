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

import ast
import copy
import csv
import dataclasses
import json
import logging
import math
import shlex
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, Optional, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import cloudai.metrics
import cloudai.report_generator.inference_report as inference_report
import cloudai.util
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
from cloudai.workloads.common import llm_serving

AIPERF_ARTIFACTS_DIR = "aiperf_artifacts"
AIPERF_ACCURACY_ARTIFACTS_DIR = "aiperf_accuracy_artifacts"
AIPERF_ACCURACY_RESULTS_CSV = "accuracy_results.csv"
LMCACHE_CONFIG_FILE_NAME = "lmcache-config.yaml"
LMCACHE_CONFIG_BACKUP_FILE_NAME = "lmcache-config.original.yaml"
HICACHE_CONFIG_FILE_NAME = "hicache-config.toml"


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
    distributed_executor_backend: Literal["mp", "ray"] | None = Field(
        default=None,
        serialization_alias="distributed-executor-backend",
        validation_alias=AliasChoices("distributed-executor-backend", "distributed_executor_backend"),
    )


class WorkerConfig(BaseModel):
    """Configuration for workers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cmd: str | None = None
    worker_initialized_regex: str | None = Field(
        default=None,
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
    nodes_per_worker: int | list[int] | None = Field(
        default=None,
        description=(
            "Number of physical nodes in one logical Slurm backend worker. When omitted, it defaults to one node "
            "per worker and preserves the legacy Slurm behavior."
        ),
        serialization_alias="nodes-per-worker",
        validation_alias=AliasChoices("nodes-per-worker", "nodes_per_worker"),
    )
    nodes: str | None = Field(default=None)

    args: WorkerBaseArgs = Field(default_factory=WorkerBaseArgs)

    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )

    @property
    def is_enabled(self) -> bool:
        """Return whether any configured trial launches this worker."""
        node_counts = self.num_nodes if isinstance(self.num_nodes, list) else [self.num_nodes]
        return not node_counts or any(count != 0 for count in node_counts)

    @model_validator(mode="after")
    def validate_enabled_worker_fields(self) -> "WorkerConfig":
        """Require launch fields unless every configured worker node count is zero."""
        if not self.is_enabled:
            return self

        missing_fields = []
        if not self.cmd:
            missing_fields.append("cmd")
        if not self.worker_initialized_regex:
            missing_fields.append("worker-initialized-regex")
        if missing_fields:
            raise ValueError(f"{', '.join(missing_fields)} must be set when num-nodes is non-zero")
        return self

    def has_extra_arg(self, option: str) -> bool:
        """Return whether an option is present in the worker's backend arguments."""
        values = [self.extra_args] if isinstance(self.extra_args, str) else self.extra_args or []
        return any(
            token == option or token.startswith(f"{option}=") for value in values for token in shlex.split(value)
        )

    def inference_args(self) -> dict[str, Any]:
        """Include numeric parallelism overrides appended to the backend command."""
        values = {key.replace("-", "_"): value for key, value in self.args.model_dump(exclude_none=True).items()}
        extra = [self.extra_args] if isinstance(self.extra_args, str) else self.extra_args or []
        tokens = [token for value in extra for token in shlex.split(value)]
        aliases = {"tp": "tensor_parallel_size", "pp": "pipeline_parallel_size", "dp": "data_parallel_size"}
        values.update({full: values[f"{short}_size"] for short, full in aliases.items() if f"{short}_size" in values})
        for token, following in zip(tokens, [*tokens[1:], ""], strict=False):
            if not token.startswith("--"):
                continue
            name, separator, value = token.partition("=")
            key = name.removeprefix("--").replace("-", "_")
            if key in (*aliases, "tp_size", "pp_size", "dp_size", *aliases.values()):
                value = value if separator else following
                canonical = aliases.get(key.removesuffix("_size"), key)
                try:
                    number = int(value)
                except ValueError:
                    number = None
                short = next(short for short, full in aliases.items() if full == canonical)
                values.update({canonical: number, key: number, f"{short}_size": number})
        return values

    @model_validator(mode="after")
    def validate_worker_topology(self) -> "WorkerConfig":
        """Validate scalar worker topology while allowing DSE lists before unrolling."""
        if not self.is_enabled:
            return self
        if isinstance(self.num_nodes, list) or isinstance(self.nodes_per_worker, list):
            return self

        nodes_per_worker = self.nodes_per_worker
        if nodes_per_worker is None:
            return self
        if nodes_per_worker < 1:
            raise ValueError("nodes_per_worker must be at least 1")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1 when nodes_per_worker is set")
        if self.num_nodes % nodes_per_worker != 0:
            raise ValueError("num_nodes must be divisible by nodes_per_worker")
        if nodes_per_worker > 1 and self.multiple_workers_per_node:
            raise ValueError("multiple_workers_per_node is incompatible with nodes_per_worker > 1")
        return self


class DCGMExporter(BaseModel):
    """Optional DCGM exporter launch configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool = False
    docker_image_url: str = Field(
        default="nvcr.io/nvidia/k8s/dcgm-exporter:4.5.2-4.8.1-distroless",
        serialization_alias="docker-image-url",
        validation_alias=AliasChoices("docker-image-url", "docker_image_url", "image-url", "image_url"),
    )
    port: int = 9401


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
    dcgm_exporter: DCGMExporter = Field(default_factory=DCGMExporter)

    decode_worker: WorkerConfig = WorkerConfig(
        cmd="python3 -m dynamo.vllm",
        worker_initialized_regex="VllmWorker.*has.been.initialized",
    )
    prefill_worker: WorkerConfig = WorkerConfig(
        cmd="python3 -m dynamo.vllm",
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


class GenAIPerf(Workload):
    """Workload configuration for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    name: str = "genai_perf"
    cmd: str = "genai-perf profile"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/genai_perf.sh")

    @property
    def installables(self) -> list[Installable]:
        return [self.script]


class AIPerf(Workload):
    """Workload configuration for aiperf benchmarking."""

    model_config = ConfigDict(extra="allow")

    name: str = "aiperf"
    cmd: str = "aiperf profile"
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/aiperf.sh")
    setup_cmd: str | None = Field(
        default=None,
        serialization_alias="setup-cmd",
        validation_alias=AliasChoices("setup-cmd", "setup_cmd"),
    )
    report_name: str = Field(
        default="aiperf_report.csv",
        serialization_alias="report-name",
        validation_alias=AliasChoices("report-name", "report_name"),
    )
    artifact_dir_name: str = Field(
        default=AIPERF_ARTIFACTS_DIR,
        serialization_alias="artifact-dir-name",
        validation_alias=AliasChoices("artifact-dir-name", "artifact_dir_name"),
    )
    health_check_between_phases: bool = Field(
        default=True,
        serialization_alias="health-check-between-phases",
        validation_alias=AliasChoices("health-check-between-phases", "health_check_between_phases"),
    )
    continue_on_phase_failure: bool = Field(
        default=False,
        serialization_alias="continue-on-phase-failure",
        validation_alias=AliasChoices("continue-on-phase-failure", "continue_on_phase_failure"),
    )
    between_phase_cmd: str | None = Field(
        default="true",
        serialization_alias="between-phase-cmd",
        validation_alias=AliasChoices("between-phase-cmd", "between_phase_cmd"),
    )

    @property
    def installables(self) -> list[Installable]:
        return [self.script]

    @model_validator(mode="after")
    def validate_extra_args(self) -> "AIPerf":
        if isinstance(self.extra_args, list):
            raise ValueError("AIPerf extra_args must be a string with explicit CLI syntax")
        return self


class AIPerfPhase(BaseModel):
    """Named AIPerf phase that overrides the base AIPerf configuration."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    cmd: str | None = None
    setup_cmd: str | None = Field(
        default=None,
        serialization_alias="setup-cmd",
        validation_alias=AliasChoices("setup-cmd", "setup_cmd"),
    )
    report_name: str | None = Field(
        default=None,
        serialization_alias="report-name",
        validation_alias=AliasChoices("report-name", "report_name"),
    )
    artifact_dir_name: str | None = Field(
        default=None,
        serialization_alias="artifact-dir-name",
        validation_alias=AliasChoices("artifact-dir-name", "artifact_dir_name"),
    )
    args: Args = Field(default_factory=Args)
    extra_args: str | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )
    between_phase_cmd: str | None = Field(
        default=None,
        serialization_alias="between-phase-cmd",
        validation_alias=AliasChoices("between-phase-cmd", "between_phase_cmd"),
    )


class AIPerfAccuracy(BaseModel):
    """Optional accuracy benchmark configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = "aiperf_accuracy"
    entrypoint: str = "aiperf profile"
    cli: str
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/accuracy.sh")
    setup_cmd: str | None = Field(
        default=None,
        serialization_alias="setup-cmd",
        validation_alias=AliasChoices("setup-cmd", "setup_cmd"),
    )
    artifact_dir_name: str = Field(
        default=AIPERF_ACCURACY_ARTIFACTS_DIR,
        serialization_alias="artifact-dir-name",
        validation_alias=AliasChoices("artifact-dir-name", "artifact_dir_name"),
    )

    @property
    def installables(self) -> list[Installable]:
        return [self.script]


class Constraints(BaseModel):
    """Constraints for validation of AI Dynamo configurations when using DSE."""

    model_config = ConfigDict(extra="forbid")

    prefill_tp_le_decode_tp: bool = True
    tp_times_pp_le_gpus_per_node: bool = True


class LMCacheController(BaseModel):
    """Optional LMCache controller process to launch on the frontend node."""

    model_config = ConfigDict(extra="forbid")

    cmd: str


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    model_config = ConfigDict(extra="forbid")

    dynamo_version: str = Field(
        default="f7e468c7e8ff0d1426db987564e60572167e8464",
        description="AI Dynamo Git commit, tag, or branch.",
    )
    docker_image_url: str
    startup_cmd: str | None = None
    startup_cmd_docker_image: str | None = None
    storage_cache_dir: Optional[str | list[str]] = Field(default="/tmp", serialization_alias="storage_cache_dir")
    dynamo: AIDynamoArgs
    hicache: dict | None = None
    lmcache: dict | None = None
    lmcache_controller: LMCacheController | None = None
    genai_perf: GenAIPerf = Field(default_factory=GenAIPerf)
    aiperf: AIPerf = Field(default_factory=AIPerf)
    aiperf_phases: list[AIPerfPhase] | None = None
    aiperf_accuracy: AIPerfAccuracy | None = None
    workloads: str = "genai_perf.sh"

    def resolve_aiperf_phase(self, phase: AIPerfPhase) -> AIPerf:
        """Apply phase overrides using the execution artifact paths."""
        base_data = self.aiperf.model_dump(by_alias=True, exclude_none=True)
        phase_data = phase.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        single_phase = self.aiperf_phases is None or len(self.aiperf_phases) == 1

        if "artifact-dir-name" not in phase_data and not single_phase:
            phase_data["artifact-dir-name"] = f"{self.aiperf.artifact_dir_name}/{phase.name}"
        if "report-name" not in phase_data and not single_phase:
            phase_data["report-name"] = f"aiperf_{phase.name}_report.csv"

        return AIPerf.model_validate(cloudai.util.deep_merge(base_data, phase_data))

    @field_validator("workloads", mode="before")
    @classmethod
    def validate_workloads(cls, v: str) -> str:
        allowed_workloads = ["genai_perf.sh", "aiperf.sh"]
        values = [w.strip() for w in v.split(",")]
        for workload in values:
            if workload not in allowed_workloads:
                raise ValueError(f"Invalid workload: {workload}. Available workloads: {allowed_workloads}")
        return ",".join(values)

    @property
    def workloads_list(self) -> list[str]:
        return [w.strip() for w in self.workloads.split(",")]

    @model_validator(mode="after")
    def validate_aiperf_phases(self) -> "AIDynamoCmdArgs":
        """Validate AIPerf phases."""
        if not self.aiperf_phases:
            return self

        seen = set()
        duplicates = set()
        for phase in self.aiperf_phases:
            if phase.name in seen:
                duplicates.add(phase.name)
            seen.add(phase.name)
        if duplicates:
            raise ValueError(f"AIPerf phase names must be unique. Duplicates: {sorted(duplicates)}")

        return self

    @property
    def installables(self) -> list[Installable]:
        return [
            *self.genai_perf.installables,
            *self.aiperf.installables,
            *(self.aiperf_accuracy.installables if self.aiperf_accuracy else []),
        ]


class AIDynamoBenchReport(llm_serving.LLMServingBenchReport):
    """Normalized AI Dynamo benchmark results used by comparison reports."""

    output_throughput: float
    request_throughput: float = math.nan
    total_token_throughput: float = math.nan

    @property
    def throughput(self) -> float:
        return self.output_throughput


class AIDynamoTestDefinition(TestDefinition, inference_report.InferenceReportProvider):
    """Test definition for AI Dynamo."""

    model_config = ConfigDict(extra="forbid")
    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    _dcgm_exporter_image: Optional[DockerImage] = None
    _startup_cmd_docker_image: DockerImage | None = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    _repo: GitRepo | None = None
    _hf_model: HFModel | None = None
    constraints: Constraints = Constraints()

    success_marker: str = "success-marker.txt"
    failure_marker: str = "failure-marker.txt"

    _TTFT_NAMES: ClassVar[tuple[str, ...]] = ("Time To First Token (ms)", "Time to First Token (ms)")
    _TPOT_NAMES: ClassVar[tuple[str, ...]] = ("Inter Token Latency (ms)",)
    _THROUGHPUT_NAMES: ClassVar[tuple[str, ...]] = ("Output Token Throughput (tokens/sec)",)
    _REQUEST_COUNT_NAMES: ClassVar[tuple[str, ...]] = ("Request Count (count)", "Request Count", "Request count")

    @staticmethod
    def _json_metric(data: Mapping[str, Any], name: str, statistic: str = "avg") -> float:
        try:
            return float(data.get(name, {}).get(statistic, math.nan))
        except (TypeError, ValueError, AttributeError):
            return math.nan

    @classmethod
    def parse_aiperf_json(cls, path: Path) -> AIDynamoBenchReport | None:
        """Read one phase's native export; absent statistics remain unavailable."""
        try:
            data = json.loads(path.read_text())
            count = int(cls._json_metric(data, "request_count"))
            latencies = {
                f"{statistic}_{metric}_ms": cls._json_metric(data, source, field)
                for metric, source in (("ttft", "time_to_first_token"), ("tpot", "inter_token_latency"))
                for statistic, field in (("mean", "avg"), ("median", "p50"), ("p99", "p99"))
            }
            return AIDynamoBenchReport(
                num_prompts=count,
                completed=count,
                max_concurrency=1,
                request_throughput=cls._json_metric(data, "request_throughput"),
                output_throughput=cls._json_metric(data, "output_token_throughput"),
                total_token_throughput=cls._json_metric(data, "total_token_throughput"),
                **latencies,
            )
        except (OSError, ValueError, OverflowError):
            logging.debug("Cannot parse AIPerf export at %s", path, exc_info=True)
            return None

    @staticmethod
    def _parse_metric_rows(csv_file: Path) -> dict[str, dict[str, str]]:
        """Parse both single-table GenAI-Perf and multi-table AIPerf CSV output."""
        metrics: dict[str, dict[str, str]] = {}
        header: list[str] | None = None
        with csv_file.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row or not any(cell.strip() for cell in row):
                    continue
                row = [cell.strip() for cell in row]
                if row[0] == "Metric":
                    header = row
                    continue
                if header is None:
                    continue
                metrics[row[0]] = {column: value for column, value in zip(header[1:], row[1:], strict=False) if value}
        return metrics

    @staticmethod
    def _find_value(
        metrics: Mapping[str, Mapping[str, str]], metric_names: tuple[str, ...], value_names: tuple[str, ...]
    ) -> float | None:
        for metric_name in metric_names:
            row = metrics.get(metric_name)
            if row is None:
                continue
            for value_name in value_names:
                value = row.get(value_name)
                if value is None:
                    continue
                try:
                    return float(value.replace(",", ""))
                except ValueError:
                    continue
        return None

    @classmethod
    def parse_benchmark_csv(cls, csv_file: Path, max_concurrency: int = 1) -> AIDynamoBenchReport | None:
        """Parse one benchmark or phase CSV without substituting another phase's output."""
        try:
            metrics = cls._parse_metric_rows(csv_file)
        except (OSError, csv.Error) as e:
            logging.debug(f"Error parsing AI Dynamo benchmark output {csv_file}: {e}")
            return None

        ttft_mean = cls._find_value(metrics, cls._TTFT_NAMES, ("avg", "mean", "Value"))
        tpot_mean = cls._find_value(metrics, cls._TPOT_NAMES, ("avg", "mean", "Value"))
        throughput = cls._find_value(metrics, cls._THROUGHPUT_NAMES, ("avg", "Value"))
        if ttft_mean is None and tpot_mean is None and throughput is None:
            return None

        request_count = cls._find_value(metrics, cls._REQUEST_COUNT_NAMES, ("avg", "Value"))
        if request_count is not None and not math.isfinite(request_count):
            return None
        request_throughput = cls._find_value(metrics, ("Request Throughput (requests/sec)",), ("avg", "Value"))
        median_ttft = cls._find_value(metrics, cls._TTFT_NAMES, ("p50", "median"))
        p99_ttft = cls._find_value(metrics, cls._TTFT_NAMES, ("p99",))
        median_tpot = cls._find_value(metrics, cls._TPOT_NAMES, ("p50", "median"))
        p99_tpot = cls._find_value(metrics, cls._TPOT_NAMES, ("p99",))
        return AIDynamoBenchReport(
            num_prompts=int(request_count or 0),
            completed=int(request_count or 0),
            mean_ttft_ms=ttft_mean if ttft_mean is not None else math.nan,
            median_ttft_ms=median_ttft if median_ttft is not None else math.nan,
            p99_ttft_ms=p99_ttft if p99_ttft is not None else math.nan,
            mean_tpot_ms=tpot_mean if tpot_mean is not None else math.nan,
            median_tpot_ms=median_tpot if median_tpot is not None else math.nan,
            p99_tpot_ms=p99_tpot if p99_tpot is not None else math.nan,
            output_throughput=throughput if throughput is not None else math.nan,
            request_throughput=request_throughput if request_throughput is not None else math.nan,
            max_concurrency=max_concurrency,
        )

    @model_validator(mode="before")
    @classmethod
    def restore_script_paths(cls, value: Any) -> Any:
        """Unwrap script paths in saved TOMLs until the shared serialization fix (#5006455)."""
        if not isinstance(value, dict):
            return value
        value = copy.deepcopy(value)
        configs = [value]
        cmd_args = value.get("cmd_args")
        if isinstance(cmd_args, dict):
            configs.extend(cmd_args.get(name) for name in ("genai_perf", "aiperf", "aiperf_accuracy"))
        for config in configs:
            script = config.get("script") if isinstance(config, dict) else None
            if not isinstance(script, dict):
                continue
            for key in ("src", "_installed_path"):
                path = script.get(key)
                if isinstance(path, str) and path.startswith(("PosixPath(", "WindowsPath(")) and path.endswith(")"):
                    script[key] = ast.literal_eval(path.split("(", 1)[1][:-1])
        return value

    @model_validator(mode="before")
    @classmethod
    def restore_repository_version(cls, value: Any) -> Any:
        """Read dumps from before the repository commit moved into cmd_args."""
        if isinstance(value, dict) and "repo" in value:
            value = dict(value)
            repo = value.pop("repo")
            value["cmd_args"] = dict(value["cmd_args"])
            value["cmd_args"].setdefault("dynamo_version", repo["commit"])
        return value

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        return [obs for report in self.inference_reports(system, tr) for obs in report.observations]

    def inference_role_configs(
        self, tr: TestRun, gpus_per_node: int | None
    ) -> tuple[inference_report.InferenceRoleConfig | None, inference_report.InferenceRoleConfig | None, bool | None]:
        prefill = self.cmd_args.dynamo.prefill_worker
        decode = self.cmd_args.dynamo.decode_worker
        shared_nodes = None
        if prefill.num_nodes and decode.num_nodes:
            shared_nodes = bool(
                set((prefill.nodes or "").split(",")) & set((decode.nodes or "").split(",")) - {""}
            ) or tr.nnodes < cast(int, prefill.num_nodes) + cast(int, decode.num_nodes)
        distributed = self.cmd_args.dynamo.backend == "sglang_dsr1"
        roles = []
        for worker in (prefill, decode):
            if not worker.num_nodes:
                roles.append(None)
                continue
            values = worker.inference_args()
            nodes = cast(int, worker.num_nodes)
            if distributed:
                default_size = nodes * gpus_per_node if gpus_per_node else None
                values["tensor_parallel_size"] = values.get("tp_size", default_size)
                values["data_parallel_size"] = values.get("data_parallel_size", values.get("dp_size", default_size))
            values.setdefault("data_parallel_size", values.get("dp_size", 1))
            role = llm_serving.inference_role_config(values)
            footprint = (role.tensor_parallel_size or 0) * (role.pipeline_parallel_size or 0)
            count: int | None = 1
            if worker.multiple_workers_per_node and not shared_nodes and not distributed:
                count = gpus_per_node // footprint if gpus_per_node and footprint else None
            nodes_per_instance = nodes if distributed else cast(int, worker.nodes_per_worker or 1)
            multinode = nodes_per_instance > 1
            roles.append(
                dataclasses.replace(
                    role,
                    num_nodes=nodes,
                    nodes_per_instance=nodes_per_instance,
                    instances_per_node=None if multinode else count,
                    num_instances=nodes // nodes_per_instance
                    if multinode
                    else (nodes * count if count is not None else None),
                )
            )
        return roles[0], roles[1], shared_nodes

    def serving_gpu_count(self, roles: tuple[inference_report.InferenceRoleConfig | None, ...]) -> int | None:
        """Count configured engine devices, including replicas; attention DP shares TP GPUs."""
        total = 0
        for role, worker in zip(
            roles, (self.cmd_args.dynamo.prefill_worker, self.cmd_args.dynamo.decode_worker), strict=True
        ):
            if role is None:
                continue
            if (
                role.tensor_parallel_size is None
                or role.pipeline_parallel_size is None
                or role.data_parallel_size is None
                or role.num_instances is None
            ):
                return None
            dp_attention = (
                self.cmd_args.dynamo.backend == "sglang_dsr1"
                or worker.has_extra_arg("--enable-dp-attention")
                or worker.inference_args().get("enable_dp_attention", False) in (True, "")
            )
            dp = 1 if self.cmd_args.dynamo.backend != "vllm" and dp_attention else role.data_parallel_size or 1
            total += role.tensor_parallel_size * (role.pipeline_parallel_size or 1) * dp * role.num_instances
        return total or None

    def parse_benchmark(self, benchmark: Workload, output_path: Path) -> AIDynamoBenchReport | None:
        csv_file = output_path / benchmark.report_name
        if isinstance(benchmark, AIPerf):
            artifacts = output_path / benchmark.artifact_dir_name
            results = self.parse_aiperf_json(artifacts / "profile_export_aiperf.json")
            if results is not None:
                return results
            if (artifacts / "profile_export_aiperf.csv").is_file():
                csv_file = artifacts / "profile_export_aiperf.csv"
        return self.parse_benchmark_csv(csv_file)

    @staticmethod
    def benchmark_observations(
        results: AIDynamoBenchReport | None,
        accuracy: float | None,
        allocated_gpus: int | None,
        serving_gpus: int | None,
        users: Any,
    ) -> list[cloudai.metrics.MetricObservation]:
        observations = llm_serving.llm_serving_metric_observations(
            results, accuracy, allocated_gpu_count=allocated_gpus
        )
        if results is None or results.completed <= 0 or not math.isfinite(results.total_token_throughput):
            return observations
        for metric, divisor in (
            (cloudai.metrics.TOTAL_TOKEN_THROUGHPUT, 1),
            (cloudai.metrics.TOTAL_TOKEN_THROUGHPUT_PER_USER, users),
            (cloudai.metrics.TOTAL_TOKEN_THROUGHPUT_PER_SERVING_GPU, serving_gpus),
        ):
            try:
                divisor = float(divisor) if divisor is not None else math.nan
            except (TypeError, ValueError):
                continue
            if math.isfinite(divisor) and divisor > 0:
                observations.append(
                    cloudai.metrics.MetricObservation(metric, results.total_token_throughput / divisor, {})
                )
        return observations

    def inference_reports(self, system: System, tr: TestRun) -> list[inference_report.InferenceReport]:
        reports = []
        gpus_per_node = getattr(system, "gpus_per_node", None)
        allocated_gpus = tr.nnodes * gpus_per_node if gpus_per_node else None
        accuracy = parse_aiperf_accuracy(tr.output_path) if self.cmd_args.aiperf_accuracy else None
        visible_gpus = self.extra_env_vars.get("CUDA_VISIBLE_DEVICES")
        available_gpus = len(str(visible_gpus).split(",")) if visible_gpus is not None else gpus_per_node
        prefill, decode, shared_nodes = self.inference_role_configs(tr, available_gpus)
        serving_gpus = self.serving_gpu_count((prefill, decode))
        for script in self.cmd_args.workloads_list:
            benchmark = self.get_workload_map()[script]
            phases = self.cmd_args.aiperf_phases if benchmark is self.cmd_args.aiperf else None
            for phase in phases or [None]:
                resolved = self.cmd_args.resolve_aiperf_phase(phase) if phase else benchmark
                dimensions = {}
                if len(self.cmd_args.workloads_list) > 1:
                    dimensions[cloudai.metrics.BENCHMARK.key] = (
                        "aiperf" if benchmark is self.cmd_args.aiperf else "genai_perf"
                    )
                if phase:
                    dimensions[cloudai.metrics.PHASE.key] = phase.name
                results = self.parse_benchmark(resolved, tr.output_path)
                args = {
                    key.replace("-", "_"): value for key, value in resolved.args.model_dump(exclude_none=True).items()
                }
                users = args.get("users", args.get("concurrency"))
                measured = self.benchmark_observations(results, accuracy, allocated_gpus, serving_gpus, users)
                observations = [
                    dataclasses.replace(observation, dimensions={**dimensions, **observation.dimensions})
                    for observation in measured
                ]
                if not observations:
                    continue
                config = inference_report.InferenceConfig(
                    model=self.cmd_args.dynamo.model,
                    backend=self.cmd_args.dynamo.backend,
                    benchmark="aiperf" if isinstance(resolved, AIPerf) else "genai_perf",
                    phase=phase.name if phase else None,
                    num_nodes=tr.nnodes,
                    gpus_per_node=gpus_per_node,
                    serving_gpu_count=serving_gpus,
                    container_image=self.cmd_args.docker_image_url,
                    input_tokens=args.get("synthetic_input_tokens_mean", args.get("input_sequence_length")),
                    output_tokens=args.get("output_tokens_mean", args.get("output_sequence_length")),
                    concurrency=args.get("concurrency"),
                    users=args.get("users"),
                    request_count=args.get("request_count"),
                    request_rate=args.get("request_rate"),
                    prefill=prefill,
                    decode=decode,
                    shared_nodes=shared_nodes,
                )
                reports.append(inference_report.InferenceReport(config=config, observations=observations))
        return reports

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
            self.cmd_args.aiperf.script.src.name: self.cmd_args.aiperf,
        }

    @property
    def repo(self) -> GitRepo:
        """Return the AI Dynamo repository selected by ``cmd_args.dynamo_version``."""
        version = self.cmd_args.dynamo_version
        if self._repo is None or self._repo.commit != version:
            self._repo = GitRepo(url="https://github.com/ai-dynamo/dynamo.git", commit=version)
        return self._repo

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def dcgm_exporter_image(self) -> DockerImage | None:
        if not self.cmd_args.dynamo.dcgm_exporter.enabled:
            return None

        image_url = self.cmd_args.dynamo.dcgm_exporter.docker_image_url
        if not self._dcgm_exporter_image or self._dcgm_exporter_image.url != image_url:
            self._dcgm_exporter_image = DockerImage(url=image_url)
        return self._dcgm_exporter_image

    @property
    def startup_cmd_docker_image(self) -> DockerImage | None:
        image_url = self.cmd_args.startup_cmd_docker_image
        if not image_url:
            return None
        if not self._startup_cmd_docker_image or self._startup_cmd_docker_image.url != image_url:
            self._startup_cmd_docker_image = DockerImage(url=image_url)
        return self._startup_cmd_docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            logging.info(f"Creating HFModel for: {self.cmd_args.dynamo.model}")
            self._hf_model = HFModel(model_name=self.cmd_args.dynamo.model)
        return self._hf_model

    @property
    def installables(self) -> list[Installable]:
        """Get all installables for this test definition."""
        installables = [
            self.docker_image,
            self.repo,
            self.script,
            self.hf_model,
            *self.cmd_args.installables,
            *super().installables,
        ]
        if self.dcgm_exporter_image:
            installables.append(self.dcgm_exporter_image)
        if self.cmd_args.startup_cmd and self.startup_cmd_docker_image:
            installables.append(self.startup_cmd_docker_image)
        return installables

    def _has_aiperf_accuracy_results(self, output_path: Path) -> bool:
        accuracy = parse_aiperf_accuracy(output_path)
        if accuracy is None:
            logging.info(f"AIPerf accuracy results not found in {output_path}.")
            return False

        logging.info(f"AIPerf accuracy results found in {output_path}: {accuracy}")
        return True

    def _was_workload_report_produced(self, output_path: Path, workload: str, workload_config: Workload) -> bool:
        report_name = workload_config.report_name
        if report_name is None:
            logging.warning(f"Workload {workload} has no report_name configured")
            return False

        workload_csv_file = output_path / report_name
        if not workload_csv_file.exists():
            logging.info(f"Result file ({workload_csv_file.absolute()}) not found for workload: {workload}")
            return False

        logging.info(f"Result file ({workload_csv_file.absolute()}) exists for {workload}")
        return True

    def _was_workload_successful(self, output_path: Path, workload: str, workload_map: dict[str, Workload]) -> bool:
        workload_config = workload_map.get(workload)
        if workload_config is None:
            logging.info(f"Workload {workload} not found in workload map")
            return False

        if not self._was_workload_report_produced(output_path, workload, workload_config):
            return False

        if isinstance(workload_config, AIPerf):
            request_count = parse_aiperf_request_count(output_path / workload_config.report_name)
            if request_count is None or request_count <= 0:
                logging.info(f"AIPerf report has no successful requests: {output_path / workload_config.report_name}")
                return False

        return True

    def _were_workloads_successful(self, output_path: Path) -> bool:
        workload_map = self.get_workload_map()
        result = True
        for workload in self.cmd_args.workloads_list:
            result = self._was_workload_successful(output_path, workload, workload_map) and result
        return result

    def _was_aiperf_accuracy_successful(self, output_path: Path) -> bool:
        if self.cmd_args.aiperf_accuracy is None:
            return True

        return self._has_aiperf_accuracy_results(output_path)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        failure_marker = output_path / self.failure_marker
        success_marker = output_path / self.success_marker

        if failure_marker.exists():
            contents = failure_marker.read_text()
            return JobStatusResult(False, error_message=f"Failure marker found:\n{contents}")

        if not success_marker.exists():
            return JobStatusResult(False, error_message=f"Success marker file not found: {success_marker.absolute()}")

        workloads_successful = self._were_workloads_successful(output_path)
        accuracy_successful = self._was_aiperf_accuracy_successful(output_path)
        return JobStatusResult(workloads_successful and accuracy_successful)

    def constraint_check(self, tr: TestRun, system: Optional[System]) -> bool:
        prefill_worker = tr.test.cmd_args.dynamo.prefill_worker
        decode_worker = tr.test.cmd_args.dynamo.decode_worker

        prefill_parallelism = (
            (int(prefill_worker.args.tensor_parallel_size), int(prefill_worker.args.pipeline_parallel_size))
            if prefill_worker.is_enabled
            else None
        )
        decode_parallelism = (
            (int(decode_worker.args.tensor_parallel_size), int(decode_worker.args.pipeline_parallel_size))
            if decode_worker.is_enabled
            else None
        )

        if (
            self.constraints.prefill_tp_le_decode_tp
            and prefill_parallelism is not None
            and decode_parallelism is not None
            and prefill_parallelism[0] > decode_parallelism[0]
        ):
            logging.info("constraint_check failed for: prefill_tp_le_decode_tp")
            return False
        logging.info("constraint_check passed for: prefill_tp_le_decode_tp")

        gpus_per_node = int(getattr(cast(SlurmSystem, system), "gpus_per_node", 0) or 0)
        role_footprints: dict[str, int] = {}
        for role, worker, parallelism in (
            ("prefill", prefill_worker, prefill_parallelism),
            ("decode", decode_worker, decode_parallelism),
        ):
            if parallelism is None:
                role_footprints[role] = 0
                continue

            tp, pp = parallelism
            num_nodes = int(worker.num_nodes)
            nodes_per_worker = int(worker.nodes_per_worker or 1)
            world_size = tp * pp
            data_parallel_size = int(worker.args.data_parallel_size or 1)
            backend = tr.test.cmd_args.dynamo.backend
            is_vllm_multinode_dp = backend == "vllm" and nodes_per_worker > 1 and data_parallel_size > 1

            if (
                backend == "sglang"
                and nodes_per_worker > 1
                and data_parallel_size > 1
                and not worker.has_extra_arg("--enable-dp-attention")
            ):
                logging.info("constraint_check failed: multinode SGLang DP requires --enable-dp-attention")
                return False
            if (
                backend == "vllm"
                and nodes_per_worker > 1
                and worker.args.distributed_executor_backend not in {None, "mp"}
            ):
                logging.info("constraint_check failed: multinode vLLM requires the mp executor")
                return False
            if (
                nodes_per_worker < 1
                or num_nodes < 1
                or num_nodes % nodes_per_worker != 0
                or world_size < 1
                or (is_vllm_multinode_dp and data_parallel_size % nodes_per_worker != 0)
                or (not is_vllm_multinode_dp and world_size % nodes_per_worker != 0)
                or (nodes_per_worker > 1 and worker.multiple_workers_per_node)
            ):
                logging.info("constraint_check failed for invalid %s worker topology", role)
                return False

            local_footprint = (
                world_size * (data_parallel_size // nodes_per_worker)
                if is_vllm_multinode_dp
                else world_size // nodes_per_worker
            )
            if gpus_per_node > 0 and self.constraints.tp_times_pp_le_gpus_per_node and local_footprint > gpus_per_node:
                logging.info("constraint_check failed for %s worker GPU capacity", role)
                return False
            role_footprints[role] = local_footprint

        logging.info("constraint_check passed for: tp_times_pp_le_gpus_per_node")

        role_total_nodes = sum(
            int(worker.num_nodes) if worker.is_enabled else 0 for worker in (prefill_worker, decode_worker)
        )
        prefill_nodes = set(prefill_worker.nodes.split(",")) if prefill_worker.nodes else set()
        decode_nodes = set(decode_worker.nodes.split(",")) if decode_worker.nodes else set()
        has_explicit_allocation = getattr(tr, "num_nodes_explicit", False) or bool(tr.nodes)
        shared_node_disagg = bool(prefill_nodes & decode_nodes) or (
            has_explicit_allocation and tr.nnodes < role_total_nodes
        )
        if (
            shared_node_disagg
            and gpus_per_node > 0
            and self.constraints.tp_times_pp_le_gpus_per_node
            and (role_footprints["prefill"] + role_footprints["decode"] > gpus_per_node)
        ):
            logging.info("constraint_check failed for: shared_node_tp_pp_sum_le_gpus_per_node")
            return False
        logging.info("constraint_check passed for: shared_node_tp_pp_sum_le_gpus_per_node")

        return True


def _parse_accuracy_value(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        accuracy = float(value)
        return accuracy / 100 if accuracy > 1 else accuracy

    raw_value = value.strip()
    if not raw_value:
        return None

    is_percentage = raw_value.endswith("%")
    if is_percentage:
        raw_value = raw_value[:-1].strip()

    try:
        accuracy = float(raw_value)
    except ValueError:
        return None

    return accuracy / 100 if is_percentage or accuracy > 1 else accuracy


def _parse_count_value(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_aiperf_request_count(report_path: Path) -> int | None:
    """Return the number of successful requests recorded by AIPerf."""
    if not report_path.exists():
        return None

    try:
        with report_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.reader(csv_file):
                if len(row) >= 2 and row[0].strip() == "Request Count":
                    request_count = _parse_count_value(row[1])
                    if request_count is None or not request_count.is_integer():
                        return None
                    return int(request_count)
    except (OSError, csv.Error):
        return None
    return None


def parse_aiperf_accuracy(output_path: Path) -> float | None:
    """
    Parse AIPerf accuracy from accuracy_results.csv.

    Expected CSV format:
        Task,Correct,Total,Accuracy
        abstract_algebra,35,100,35.00%
        OVERALL,8368,14042,59.59%

    AIPerf writes this file under aiperf_artifacts; CloudAI's wrapper also copies
    it to the run output directory when present. The returned value is normalized
    to a 0.0-1.0 fraction.
    """
    candidates = [
        output_path / AIPERF_ACCURACY_RESULTS_CSV,
        output_path / AIPERF_ACCURACY_ARTIFACTS_DIR / AIPERF_ACCURACY_RESULTS_CSV,
        output_path / AIPERF_ARTIFACTS_DIR / AIPERF_ACCURACY_RESULTS_CSV,
    ]

    for csv_file in candidates:
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            continue

        fallback_accuracy: float | None = None
        with csv_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                accuracy = _parse_accuracy_value(row.get("Accuracy") or row.get("accuracy") or row.get("Value"))
                if accuracy is None:
                    correct = _parse_count_value(row.get("Correct") or row.get("correct"))
                    total = _parse_count_value(row.get("Total") or row.get("total"))
                    if correct is not None and total:
                        accuracy = correct / total
                if accuracy is None:
                    continue

                task = (row.get("Task") or row.get("task") or row.get("Metric") or "").strip().upper()
                if task == "OVERALL":
                    return accuracy
                if fallback_accuracy is None:
                    fallback_accuracy = accuracy

        if fallback_accuracy is not None:
            return fallback_accuracy

    return None
