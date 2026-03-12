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

from typing import ClassVar, cast

from rich.console import Console
from rich.table import Table

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

from .sglang import SGLANG_BENCH_JSONL_FILE, SglangTestDefinition, parse_sglang_bench_output
from .slurm_command_gen_strategy import sglang_all_gpu_ids


class SGLangBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Generate report and metrics for SGLang benchmark output."""

    metrics: ClassVar[list[str]] = [
        "default",
        "throughput",
        "tps-per-user",
        "tps-per-gpu",
    ]

    def can_handle_directory(self) -> bool:
        return parse_sglang_bench_output(self.test_run.output_path / SGLANG_BENCH_JSONL_FILE) is not None

    def used_gpus_count(self) -> int:
        return len(
            sglang_all_gpu_ids(
                cast(SglangTestDefinition, self.test_run.test), getattr(self.system, "gpus_per_node", None)
            )
        )

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        results = parse_sglang_bench_output(self.test_run.output_path / SGLANG_BENCH_JSONL_FILE)
        if results is None:
            return METRIC_ERROR

        if metric == "tps-per-user":
            return results.tps_per_user if results.tps_per_user is not None else METRIC_ERROR
        if metric == "tps-per-gpu":
            used_gpus = self.used_gpus_count()
            return results.throughput / used_gpus

        return results.throughput

    def generate_report(self) -> None:
        results = parse_sglang_bench_output(self.test_run.output_path / SGLANG_BENCH_JSONL_FILE)
        if results is None:
            return

        console = Console()
        table = Table(title=f"SGLang Benchmark Results ({self.test_run.output_path})", title_justify="left")
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
