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

import json
import logging
from functools import cache
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.workloads.vllm.slurm_command_gen_strategy import vllm_all_gpu_ids

from .vllm import VLLM_BENCH_JSON_FILE, VllmTestDefinition


class VLLMBenchReport(BaseModel):
    """Report for vLLM benchmark results."""

    model_config = ConfigDict(extra="ignore")

    num_prompts: int
    completed: int
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    output_throughput: float
    max_concurrency: int

    @property
    def throughput(self) -> float:
        return self.output_throughput

    @property
    def concurrency(self) -> int:
        return self.max_concurrency

    @property
    def tps_per_user(self) -> float | None:
        if self.concurrency <= 0:
            return None
        return self.throughput / self.concurrency


@cache
def parse_vllm_bench_output(res_file: Path) -> VLLMBenchReport | None:
    """Parse the vLLM benchmark output file and return a VLLMBenchReport object."""
    if not res_file.is_file():
        return None

    try:
        data = json.loads(res_file.read_text())
        return VLLMBenchReport.model_validate(data)
    except Exception as e:
        logging.debug(f"Error parsing vLLM benchmark output: {e}")
        return None


class VLLMBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Generate a report for vLLM benchmark results."""

    metrics: ClassVar[list[str]] = [
        "default",
        "throughput",
        "tps-per-user",
        "tps-per-gpu",
    ]

    def can_handle_directory(self) -> bool:
        return parse_vllm_bench_output(self.test_run.output_path / VLLM_BENCH_JSON_FILE) is not None

    def used_gpus_count(self) -> int:
        return len(
            vllm_all_gpu_ids(cast(VllmTestDefinition, self.test_run.test), getattr(self.system, "gpus_per_node", None))
        )

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        results = parse_vllm_bench_output(self.test_run.output_path / VLLM_BENCH_JSON_FILE)
        if results is None:
            return METRIC_ERROR

        if metric in ("default", "throughput"):
            return results.throughput
        if metric == "tps-per-user":
            return results.tps_per_user if results.tps_per_user is not None else METRIC_ERROR
        if metric == "tps-per-gpu":
            used_gpus = self.used_gpus_count()
            return results.throughput / used_gpus

        return METRIC_ERROR

    def generate_report(self) -> None:
        results = parse_vllm_bench_output(self.test_run.output_path / VLLM_BENCH_JSON_FILE)
        if results is None:
            return

        console = Console()
        table = Table(title=f"vLLM Benchmark Results ({self.test_run.output_path})", title_justify="left")
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
