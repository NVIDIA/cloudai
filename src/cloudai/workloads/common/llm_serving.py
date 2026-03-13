# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.table import Table
from typing_extensions import Self

from cloudai.core import METRIC_ERROR, DockerImage, HFModel, Installable, ReportGenerationStrategy
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmCommandGenStrategy

TestDefT = TypeVar("TestDefT")
ReportT = TypeVar("ReportT", bound="LLMServingBenchReport")
LLMServingArgsT = TypeVar("LLMServingArgsT", bound="LLMServingArgs")
LLMServingCmdArgsT = TypeVar("LLMServingCmdArgsT", bound="LLMServingCmdArgs")


def all_gpu_ids(tdef: LLMServingTestDefinition[LLMServingCmdArgsT], system_gpus_per_node: int | None) -> list[int]:
    cuda_devices = str(tdef.extra_env_vars.get("CUDA_VISIBLE_DEVICES", ""))
    if (tdef.cmd_args.prefill and tdef.cmd_args.prefill.gpu_ids) and tdef.cmd_args.decode.gpu_ids:
        cuda_devices = f"{tdef.cmd_args.prefill.gpu_ids},{tdef.cmd_args.decode.gpu_ids}"
    if cuda_devices:
        return [int(gpu_id) for gpu_id in cuda_devices.split(",")]
    return list(range(system_gpus_per_node or 1))


class LLMServingArgs(CmdArgs):
    """Shared serve-argument serialization for LLM serving workloads."""

    gpu_ids: str | list[str] | None = None

    @property
    def serve_args_exclude(self) -> set[str]:
        """Fields consumed internally and excluded from generic serve args."""
        return set()

    @property
    def serve_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.model_dump(exclude=self.serve_args_exclude, exclude_none=True).items():
            opt = f"--{key.replace('_', '-')}"
            if value == "":
                args.append(opt)
            else:
                args.extend([opt, str(value)])
        return args


class LLMServingCmdArgs(CmdArgs, Generic[LLMServingArgsT]):
    """Shared command-argument shape for LLM serving workloads."""

    docker_image_url: str
    model: str
    serve_wait_seconds: int = 300
    prefill: LLMServingArgsT | None = Field(default=None)
    decode: LLMServingArgsT


class LLMServingTestDefinition(TestDefinition, Generic[LLMServingCmdArgsT]):
    """Shared test-definition behavior for LLM serving workloads."""

    cmd_args: LLMServingCmdArgsT
    _docker_image: DockerImage | None = None
    _hf_model: HFModel | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            self._hf_model = HFModel(model_name=self.cmd_args.model)
        return self._hf_model

    @property
    def extra_installables(self) -> list[Installable]:
        return []

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image, self.hf_model, *self.extra_installables]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._docker_image = None
        self._hf_model = None

    @model_validator(mode="after")
    def check_gpu_ids_setup(self) -> Self:
        if self.cmd_args.prefill:
            prefill_set = bool(self.cmd_args.prefill.gpu_ids)
            decode_set = bool(self.cmd_args.decode.gpu_ids)
            if prefill_set != decode_set:
                raise ValueError("Both prefill and decode gpu_ids must be set or both must be None.")
        return self


class LLMServingBenchReport(BaseModel, ABC):
    """Shared benchmark result shape for LLM serving workloads."""

    model_config = ConfigDict(extra="ignore")

    num_prompts: int
    completed: int
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    max_concurrency: int

    @property
    @abstractmethod
    def throughput(self) -> float:
        """Workload-specific throughput metric."""

    @property
    def concurrency(self) -> int:
        return self.max_concurrency

    @property
    def tps_per_user(self) -> float | None:
        if self.concurrency <= 0:
            return None
        return self.throughput / self.concurrency


class LLMServingReportGenerationStrategy(ReportGenerationStrategy, Generic[TestDefT, ReportT], ABC):
    """Shared report generation strategy for LLM serving workloads."""

    metrics: ClassVar[list[str]] = [
        "default",
        "throughput",
        "tps-per-user",
        "tps-per-gpu",
    ]

    @property
    @abstractmethod
    def result_file_name(self) -> str:
        """Benchmark result file name for this workload."""

    @property
    @abstractmethod
    def report_title(self) -> str:
        """User-facing report title."""

    @abstractmethod
    def parse_output(self, path: Path) -> ReportT | None:
        """Parse benchmark output into a report model."""

    @abstractmethod
    def all_gpu_ids(self, tdef: TestDefT, gpus_per_node: int | None) -> list[int]:
        """Return GPU ids used by this workload."""

    def parse_results(self) -> ReportT | None:
        return self.parse_output(self.test_run.output_path / self.result_file_name)

    def can_handle_directory(self) -> bool:
        return self.parse_results() is not None

    def used_gpus_count(self) -> int:
        return len(self.all_gpu_ids(cast(TestDefT, self.test_run.test), getattr(self.system, "gpus_per_node", None)))

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        results = self.parse_results()
        if results is None:
            return METRIC_ERROR

        if metric == "tps-per-user":
            return results.tps_per_user if results.tps_per_user is not None else METRIC_ERROR
        if metric == "tps-per-gpu":
            return results.throughput / self.used_gpus_count()

        return results.throughput

    def generate_report(self) -> None:
        results = self.parse_results()
        if results is None:
            return

        console = Console()
        table = Table(title=f"{self.report_title} ({self.test_run.output_path})", title_justify="left")
        table.add_column("Successful prompts", justify="right")
        table.add_column("TTFT Mean, ms", justify="right")
        table.add_column("TTFT Median, ms", justify="right")
        table.add_column("TTFT P99, ms", justify="right")
        table.add_column("TPOT Mean, ms", justify="right")
        table.add_column("TPOT Median, ms", justify="right")
        table.add_column("TPOT P99, ms", justify="right")
        table.add_row(
            f"{results.completed / results.num_prompts * 100:.2f}% ({results.completed} of {results.num_prompts})",
            f"{results.mean_ttft_ms:.4f}",
            f"{results.median_ttft_ms:.4f}",
            f"{results.p99_ttft_ms:.4f}",
            f"{results.mean_tpot_ms:.4f}",
            f"{results.median_tpot_ms:.4f}",
            f"{results.p99_tpot_ms:.4f}",
        )
        console.print(table)


class LLMServingSlurmCommandGenStrategy(SlurmCommandGenStrategy, Generic[LLMServingCmdArgsT], ABC):
    """Shared Slurm helpers for LLM serving workloads."""

    @property
    @abstractmethod
    def tdef(self) -> LLMServingTestDefinition[LLMServingCmdArgsT]:
        """Typed access to the workload test definition."""

    def _container_mounts(self) -> list[str]:
        return [f"{self.system.hf_home_path.absolute()}:/root/.cache/huggingface"]

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    @property
    def gpu_ids(self) -> list[int]:
        return all_gpu_ids(self.tdef, self.system.gpus_per_node)

    @property
    def prefill_gpu_ids(self) -> list[int]:
        if self.tdef.cmd_args.prefill and self.tdef.cmd_args.prefill.gpu_ids:
            return [int(gpu_id) for gpu_id in str(self.tdef.cmd_args.prefill.gpu_ids).split(",")]
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[:mid]

    @property
    def decode_gpu_ids(self) -> list[int]:
        if self.tdef.cmd_args.decode.gpu_ids:
            return [int(gpu_id) for gpu_id in str(self.tdef.cmd_args.decode.gpu_ids).split(",")]
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[mid:]

    def generate_wait_for_health_function(self) -> str:
        timeout = self.tdef.cmd_args.serve_wait_seconds
        return f"""\
wait_for_health() {{
    local endpoint="$1"
    local timeout={timeout}
    local interval=5
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if curl -sf "$endpoint" > /dev/null 2>&1; then
            echo "Health check passed: $endpoint"
            return 0
        fi
        sleep "$interval"
    done

    echo "Timeout waiting for: $endpoint"
    return 1
}}"""

    def _gen_llm_serving_srun_command(
        self,
        serve_commands: list[list[str]],
        bench_cmd: str,
        health_func: str,
    ) -> str:
        srun_prefix = " ".join(self.gen_srun_prefix())
        if len(serve_commands) == 1:
            return self._gen_aggregated_script(srun_prefix, serve_commands[0], bench_cmd, health_func)
        return self._gen_disaggregated_script(srun_prefix, serve_commands, bench_cmd, health_func)

    @abstractmethod
    def _gen_aggregated_script(self, srun_prefix: str, serve_cmd: list[str], bench_cmd: str, health_func: str) -> str:
        """Render the aggregated-mode srun script."""

    @abstractmethod
    def _gen_disaggregated_script(
        self, srun_prefix: str, serve_commands: list[list[str]], bench_cmd: str, health_func: str
    ) -> str:
        """Render the disaggregated-mode srun script."""
