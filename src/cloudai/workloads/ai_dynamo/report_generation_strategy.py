# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import shutil
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.lazy_imports import lazy


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    metrics: ClassVar[list[str]] = [
        "default",
        "output-token-throughput",
        "request-throughput",
        "time-to-first-token",
        "time-to-second-token",
        "request-latency",
        "inter-token-latency",
    ]

    def can_handle_directory(self) -> bool:
        output_path = self.test_run.output_path
        csv_files = list(output_path.rglob("profile_genai_perf.csv"))
        json_files = list(output_path.rglob("profile_genai_perf.json"))
        return len(csv_files) > 0 and len(json_files) > 0

    def _read_metric_from_csv(self, metric_name: str) -> float:
        output_path = self.test_run.output_path
        source_csv = next(output_path.rglob("profile_genai_perf.csv"))

        if source_csv.stat().st_size == 0:
            return METRIC_ERROR

        df = lazy.pd.read_csv(source_csv)
        metric_row = df[df["Metric"] == metric_name]

        if metric_row.empty:
            return METRIC_ERROR

        if "Value" in df.columns and not metric_row["Value"].empty:
            return float(metric_row["Value"].iloc[0])

        if "avg" in df.columns and not metric_row["avg"].empty:
            return float(metric_row["avg"].iloc[0].replace(",", ""))

        return METRIC_ERROR

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        metric_mapping = {
            "default": "Output Token Throughput (tokens/sec)",
            "output-token-throughput": "Output Token Throughput (tokens/sec)",
            "request-throughput": "Request Throughput (per sec)",
            "time-to-first-token": "Time To First Token (ms)",
            "time-to-second-token": "Time To Second Token (ms)",
            "request-latency": "Request Latency (ms)",
            "inter-token-latency": "Inter Token Latency (ms)",
        }

        mapped_metric = metric_mapping.get(metric)
        if not mapped_metric:
            return METRIC_ERROR

        return self._read_metric_from_csv(mapped_metric)

    def generate_report(self) -> None:
        output_path = self.test_run.output_path
        source_csv = next(output_path.rglob("profile_genai_perf.csv"))
        target_csv = output_path / "report.csv"

        shutil.copy2(source_csv, target_csv)

        gpus_per_node = None
        if isinstance(self.system, SlurmSystem):
            gpus_per_node = self.system.gpus_per_node

        if gpus_per_node is None:
            logging.warning("gpus_per_node is None, skipping Overall Output Tokens per Second per GPU calculation.")
            return

        num_frontend_nodes = 1
        num_prefill_nodes = self.test_run.test.test_definition.cmd_args.dynamo.prefill_worker.num_nodes
        num_decode_nodes = self.test_run.test.test_definition.cmd_args.dynamo.decode_worker.num_nodes

        total_gpus = (num_frontend_nodes + num_prefill_nodes + num_decode_nodes) * gpus_per_node

        with open(source_csv, "r") as f:
            lines = f.readlines()
            output_token_throughput_line = next(
                (line for line in lines if "Output Token Throughput (tokens/sec)" in line), None
            )
            if output_token_throughput_line:
                output_token_throughput = float(output_token_throughput_line.split(",")[1].strip())

                overall_output_tokens_per_second_per_gpu = output_token_throughput / total_gpus

                with open(target_csv, "a") as f:
                    f.write(f"Overall Output Tokens per Second per GPU,{overall_output_tokens_per_second_per_gpu}\n")
