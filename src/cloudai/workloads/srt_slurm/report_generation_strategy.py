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

import csv
import logging
from typing import ClassVar, cast

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy

from .results import load_results
from .srt_slurm import SRTSlurmTestDefinition


class SRTSlurmReportGenerationStrategy(ReportGenerationStrategy):
    """Write every concurrency; use the best throughput point for scalar DSE metrics."""

    metrics: ClassVar[list[str]] = [
        "default",
        "output_throughput",
        "throughput_toks",
        "total_token_throughput",
        "request_throughput",
        "tput_per_gpu",
        "input_tput_per_gpu",
        "output_tput_per_gpu",
        "completed_requests",
        *[
            f"{family}_{stat}_ms"
            for family in ("ttft", "tpot", "itl", "e2el")
            for stat in ("mean", "median", "p75", "p90", "p95", "p99")
        ],
        *[
            f"{stat}_{family}"
            for family in ("ttft", "tpot", "itl", "e2el", "intvty")
            for stat in ("mean", "median", "p75", "p90", "p95", "p99")
        ],
    ]

    def results(self) -> list[dict[str, float]]:
        try:
            return load_results(cast(SRTSlurmTestDefinition, self.test_run.test), self.test_run)
        except ValueError as error:
            logging.warning("Unable to read srt-slurm results: %s", error)
            return []

    def can_handle_directory(self) -> bool:
        return bool(self.results())

    def generate_report(self) -> None:
        results = self.results()
        if not results:
            return
        fields = sorted({key for run in results for key in run})
        with (self.test_run.output_path / "srt_slurm_results.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    def get_metric(self, metric: str) -> MetricValue:
        results = self.results()
        if not results:
            return METRIC_ERROR
        key = "output_throughput" if metric == "default" else metric
        # All metrics come from the same point. AgentX already supplies the
        # serving-GPU-normalized throughput; sa-bench keeps the original default.
        best = max(results, key=lambda run: run.get("tput_per_gpu", run.get("output_throughput", 0)))
        return best.get(key, METRIC_ERROR)
