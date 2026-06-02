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

import csv
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

AIPERF_ARTIFACTS_DIR = "aiperf_artifacts"
AIPERF_ACCURACY_ARTIFACTS_DIR = "aiperf_accuracy_artifacts"
AIPERF_ACCURACY_RESULTS_CSV = "accuracy_results.csv"
LMCACHE_CONFIG_FILE_NAME = "lmcache-config.yaml"
LMCACHE_CONFIG_BACKUP_FILE_NAME = "lmcache-config.original.yaml"


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

    docker_image_url: str
    storage_cache_dir: Optional[str | list[str]] = Field(default="/tmp", serialization_alias="storage_cache_dir")
    dynamo: AIDynamoArgs
    lmcache: dict | None = None
    lmcache_controller: LMCacheController | None = None
    genai_perf: GenAIPerf = Field(default_factory=GenAIPerf)
    aiperf: AIPerf = Field(default_factory=AIPerf)
    aiperf_phases: list[AIPerfPhase] | None = None
    aiperf_accuracy: AIPerfAccuracy | None = None
    workloads: str = "genai_perf.sh"

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


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    model_config = ConfigDict(extra="forbid")
    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    _dcgm_exporter_image: Optional[DockerImage] = None
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
            self.cmd_args.aiperf.script.src.name: self.cmd_args.aiperf,
        }

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
        ]
        if self.dcgm_exporter_image:
            installables.append(self.dcgm_exporter_image)
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

        return self._was_workload_report_produced(output_path, workload, workload_config)

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
