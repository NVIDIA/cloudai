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

import csv
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm.slurm_system import SlurmSystem

if TYPE_CHECKING:
    from .ai_dynamo import AIDynamoTestDefinition

CSV_FILES_PATTERN = "profile*_genai_perf.csv"
JSON_FILES_PATTERN = "profile*_genai_perf.json"


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
        csv_files = list(output_path.rglob(CSV_FILES_PATTERN))
        json_files = list(output_path.rglob(JSON_FILES_PATTERN))
        logging.debug(f"Found CSV files: {csv_files}, JSON files: {json_files}")
        return len(csv_files) > 0 and len(json_files) > 0

    def _find_csv_file(self) -> Path | None:
        output_path = self.test_run.output_path
        if not output_path.exists() or not output_path.is_dir():
            return None

        csv_files = list(output_path.rglob(CSV_FILES_PATTERN))
        if not csv_files or csv_files[0].stat().st_size == 0:
            return None

        return csv_files[0]

    def _extract_metric_value(self, header: list[str], row: list[str], metric_idx: int) -> float | None:
        if "Value" in header:
            value_idx = header.index("Value")
            return float(row[value_idx].replace(",", ""))
        elif "avg" in header:
            avg_idx = header.index("avg")
            return float(row[avg_idx].replace(",", ""))
        return None

    def _find_metric_in_section(self, section: list[list[str]], metric_name: str) -> float | None:
        if not section:
            return None

        header = section[0]
        if "Metric" not in header:
            return None

        metric_idx = header.index("Metric")
        for row in section[1:]:
            if row[metric_idx] == metric_name:
                return self._extract_metric_value(header, row, metric_idx)
        return None

    def _read_metric_from_csv(self, metric_name: str) -> float:
        source_csv = self._find_csv_file()
        if not source_csv:
            return METRIC_ERROR

        sections = self._read_csv_sections(source_csv)
        for section in sections:
            value = self._find_metric_in_section(section, metric_name)
            if value is not None:
                return value

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

    def _calculate_total_gpus(self) -> int | None:
        gpus_per_node = None
        if isinstance(self.system, (SlurmSystem, KubernetesSystem)):
            gpus_per_node = self.system.gpus_per_node

        if gpus_per_node is None:
            return None

        tdef = cast("AIDynamoTestDefinition", self.test_run.test)

        num_frontend_nodes = 1
        num_prefill_nodes = (
            cast(int, tdef.cmd_args.dynamo.prefill_worker.num_nodes) if tdef.cmd_args.dynamo.prefill_worker else 0
        )
        num_decode_nodes = cast(int, tdef.cmd_args.dynamo.decode_worker.num_nodes)
        return (num_frontend_nodes + num_prefill_nodes + num_decode_nodes) * gpus_per_node

    def _read_csv_sections(self, source_csv: Path) -> list[list[list[str]]]:
        sections = []
        current_section = []

        with open(source_csv, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if not any(row):  # Empty row indicates section break
                    if current_section:
                        sections.append(current_section)
                        current_section = []
                else:
                    current_section.append(row)
            if current_section:
                sections.append(current_section)

        return sections

    def _write_sections_with_metric(
        self, target_csv: Path, sections: list[list[list[str]]], total_gpus: int | None
    ) -> None:
        with open(target_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # Write first section (statistical metrics)
            if sections:
                for row in sections[0]:
                    writer.writerow(row)
                writer.writerow([])  # Empty row for section break

            # Write second section with additional metric if total_gpus is available
            if len(sections) > 1:
                for row in sections[1]:
                    writer.writerow(row)
                    if total_gpus and row and row[0] == "Output Token Throughput (tokens/sec)":
                        throughput = float(row[1].replace(",", ""))
                        per_gpu_throughput = throughput / total_gpus
                        writer.writerow(["Overall Output Tokens per Second per GPU", per_gpu_throughput])
                writer.writerow([])  # Empty row for section break

            # Write remaining sections
            for section in sections[2:]:
                for row in section:
                    writer.writerow(row)
                writer.writerow([])  # Empty row for section break

    def generate_report(self) -> None:
        output_path = self.test_run.output_path
        source_csv = next(output_path.rglob(CSV_FILES_PATTERN))
        target_csv = output_path / "report.csv"

        total_gpus = self._calculate_total_gpus()
        if total_gpus is None:
            logging.warning("gpus_per_node is None, skipping Overall Output Tokens per Second per GPU calculation.")
            shutil.copy2(source_csv, target_csv)
            return

        sections = self._read_csv_sections(source_csv)
        self._write_sections_with_metric(target_csv, sections, total_gpus)
