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
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

if TYPE_CHECKING:
    from cloudai.workloads.sglang.sglang import SglangTestDefinition
    from cloudai.workloads.vllm.vllm import VllmTestDefinition

TestDefT = TypeVar("TestDefT")
ReportT = TypeVar("ReportT", bound="LLMServingBenchReport")


def all_gpu_ids(tdef: VllmTestDefinition | SglangTestDefinition, system_gpus_per_node: int | None) -> list[int]:
    cuda_devices = str(tdef.extra_env_vars.get("CUDA_VISIBLE_DEVICES", ""))
    if (tdef.cmd_args.prefill and tdef.cmd_args.prefill.gpu_ids) and tdef.cmd_args.decode.gpu_ids:
        cuda_devices = f"{tdef.cmd_args.prefill.gpu_ids},{tdef.cmd_args.decode.gpu_ids}"
    if cuda_devices:
        return [int(gpu_id) for gpu_id in cuda_devices.split(",")]
    return list(range(system_gpus_per_node or 1))


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
