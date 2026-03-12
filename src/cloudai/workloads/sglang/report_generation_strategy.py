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

from functools import cache
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

from .sglang import SGLANG_BENCH_JSONL_FILE, SglangTestDefinition, parse_sglang_bench_jsonl
from .slurm_command_gen_strategy import sglang_all_gpu_ids


class SGLangBenchReport(BaseModel):
    """Parsed benchmark data from SGLang bench_serving output."""

    successful_requests: int
    request_throughput: float
    max_concurrency: int
    mean_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None

    @property
    def throughput(self) -> float:
        return self.request_throughput

    @property
    def concurrency(self) -> int:
        return self.max_concurrency

    @property
    def tps_per_user(self) -> float | None:
        if self.concurrency <= 0:
            return None
        return self.throughput / self.concurrency


@cache
def parse_sglang_bench_output(jsonl_file: Path, default_concurrency: int) -> SGLangBenchReport | None:
    """Parse SGLang benchmark output from JSONL file."""
    summary = parse_sglang_bench_jsonl(jsonl_file)
    if summary is None:
        return None

    if summary.completed is None or summary.completed <= 0 or summary.request_throughput is None:
        return None

    return SGLangBenchReport(
        successful_requests=summary.completed,
        request_throughput=summary.request_throughput,
        max_concurrency=summary.max_concurrency or default_concurrency,
        mean_ttft_ms=summary.mean_ttft_ms,
        mean_tpot_ms=summary.mean_tpot_ms,
    )


class SGLangBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Generate report and metrics for SGLang benchmark output."""

    metrics: ClassVar[list[str]] = [
        "default",
        "throughput",
        "tps-per-user",
        "tps-per-gpu",
    ]

    def _parse(self) -> SGLangBenchReport | None:
        tdef = cast(SglangTestDefinition, self.test_run.test)
        return parse_sglang_bench_output(
            self.test_run.output_path / SGLANG_BENCH_JSONL_FILE,
            default_concurrency=tdef.bench_cmd_args.max_concurrency,
        )

    def can_handle_directory(self) -> bool:
        return self._parse() is not None

    def used_gpus_count(self) -> int:
        return len(
            sglang_all_gpu_ids(
                cast(SglangTestDefinition, self.test_run.test), getattr(self.system, "gpus_per_node", None)
            )
        )

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        results = self._parse()
        if results is None:
            return METRIC_ERROR

        if metric == "tps-per-user":
            return results.tps_per_user if results.tps_per_user is not None else METRIC_ERROR
        if metric == "tps-per-gpu":
            used_gpus = self.used_gpus_count()
            return results.throughput / used_gpus

        return results.throughput

    def generate_report(self) -> None:
        results = self._parse()
        if results is None:
            return

        console = Console()
        table = Table(title=f"SGLang Benchmark Results ({self.test_run.output_path})", title_justify="left")
        table.add_column("Successful requests", justify="right")
        table.add_column("Request throughput (req/s)", justify="right")
        table.add_column("Concurrency", justify="right")
        table.add_column("Mean TTFT (ms)", justify="right")
        table.add_column("Mean TPOT (ms)", justify="right")

        table.add_row(
            str(results.successful_requests),
            f"{results.request_throughput:.4f}",
            str(results.max_concurrency),
            f"{results.mean_ttft_ms:.4f}" if results.mean_ttft_ms is not None else "-",
            f"{results.mean_tpot_ms:.4f}" if results.mean_tpot_ms is not None else "-",
        )
        console.print(table)
