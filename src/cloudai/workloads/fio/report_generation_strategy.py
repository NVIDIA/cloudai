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

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy

_OP_RE = re.compile(r"^\s*(read|write|trim):\s+IOPS=(?P<iops>[^,]+),\s+BW=(?P<bw>[\d.]+)(?P<bw_unit>[A-Za-z/]+)")
_LAT_RE = re.compile(r"^\s+(?P<kind>clat|lat)\s+\((?P<unit>[^)]+)\):.*\bavg=(?P<avg>[\d.]+)")


@dataclass(frozen=True)
class FioSummary:
    """Summary metrics parsed from fio stdout."""

    operation: str
    iops: float
    bw: float
    bw_unit: str
    latency_avg: float | None = None
    latency_unit: str | None = None


def _parse_scaled_number(value: str) -> float:
    normalized = value.strip()
    suffix = normalized[-1:].lower()
    multiplier = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}.get(suffix, 1)
    if multiplier != 1:
        normalized = normalized[:-1]
    return float(normalized) * multiplier


def extract_fio_data(stdout_file: Path) -> list[FioSummary]:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return []

    summaries: list[FioSummary] = []
    pending_operation: str | None = None
    pending_iops: float | None = None
    pending_bw: float | None = None
    pending_bw_unit: str | None = None
    best_latency: tuple[str, float, str] | None = None

    def flush_pending() -> None:
        nonlocal pending_operation, pending_iops, pending_bw, pending_bw_unit, best_latency
        if pending_operation is None or pending_iops is None or pending_bw is None or pending_bw_unit is None:
            return
        summaries.append(
            FioSummary(
                operation=pending_operation,
                iops=pending_iops,
                bw=pending_bw,
                bw_unit=pending_bw_unit,
                latency_avg=best_latency[1] if best_latency else None,
                latency_unit=best_latency[2] if best_latency else None,
            )
        )
        pending_operation = None
        pending_iops = None
        pending_bw = None
        pending_bw_unit = None
        best_latency = None

    for line in stdout_file.read_text().splitlines():
        if match := _OP_RE.match(line):
            flush_pending()
            pending_operation = match.group(1)
            pending_iops = _parse_scaled_number(match.group("iops"))
            pending_bw = float(match.group("bw"))
            pending_bw_unit = match.group("bw_unit")
            continue

        if pending_operation is None:
            continue

        if match := _LAT_RE.match(line):
            candidate = (match.group("kind"), float(match.group("avg")), match.group("unit"))
            if best_latency is None or candidate[0] == "lat":
                best_latency = candidate

    flush_pending()
    return summaries


class FioReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for fio."""

    metrics: ClassVar[list[str]] = ["default", "read_bw", "write_bw", "read_iops", "write_iops", "read_latency"]

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        return bool(extract_fio_data(self.results_file))

    def get_metric(self, metric: str) -> MetricValue:
        rows = extract_fio_data(self.results_file)
        if not rows:
            return METRIC_ERROR

        by_operation = {row.operation: row for row in rows}
        if metric == "default":
            return next((row.bw for row in rows), METRIC_ERROR)
        if metric == "read_bw" and "read" in by_operation:
            return by_operation["read"].bw
        if metric == "write_bw" and "write" in by_operation:
            return by_operation["write"].bw
        if metric == "read_iops" and "read" in by_operation:
            return by_operation["read"].iops
        if metric == "write_iops" and "write" in by_operation:
            return by_operation["write"].iops
        if metric == "read_latency" and "read" in by_operation and by_operation["read"].latency_avg is not None:
            return by_operation["read"].latency_avg
        return METRIC_ERROR

    def generate_report(self) -> None:
        rows = extract_fio_data(self.results_file)
        if not rows:
            return

        with (self.test_run.output_path / "fio_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["operation", "iops", "bw", "bw_unit", "latency_avg", "latency_unit"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
