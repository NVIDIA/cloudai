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
import dataclasses
import logging
import pathlib
import re
import statistics
from typing import ClassVar

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy

_OP_RE = re.compile(r"^\s*(read|write|trim):\s+IOPS=(?P<iops>[^,]+),\s+BW=(?P<bw>[\d.]+)(?P<bw_unit>[A-Za-z/]+)")
_LAT_RE = re.compile(r"^\s+(?P<kind>clat|lat)\s+\((?P<unit>[^)]+)\):.*\bavg=(?P<avg>[\d.]+)")


@dataclasses.dataclass(frozen=True)
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


def extract_fio_data(stdout_file: pathlib.Path) -> list[FioSummary]:
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


def _filter_rows(rows: list[FioSummary], operation: str) -> list[FioSummary]:
    if operation == "first":
        return rows[:1]
    if operation == "all":
        return rows
    return [row for row in rows if row.operation == operation]


def _metric_values(rows: list[FioSummary], metric_name: str) -> list[float]:
    if metric_name == "bw":
        return [row.bw for row in rows]
    if metric_name == "iops":
        return [row.iops for row in rows]
    if metric_name == "latency":
        return [row.latency_avg for row in rows if row.latency_avg is not None]
    return []


def _aggregate_values(values: list[float], aggregate: str) -> MetricValue:
    if not values:
        return METRIC_ERROR
    if aggregate == "sum":
        return sum(values)
    if aggregate == "mean":
        return statistics.fmean(values)
    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "first":
        return values[0]

    logging.warning(f"Unsupported fio metric aggregate {aggregate!r}.")
    return METRIC_ERROR


class FioReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for fio."""

    metrics: ClassVar[list[str]] = ["default", "read_bw", "write_bw", "read_iops", "write_iops", "read_latency"]

    @property
    def results_file(self) -> pathlib.Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        return bool(extract_fio_data(self.results_file))

    def _configured_default_metric(self, rows: list[FioSummary]) -> MetricValue:
        cmd_args = self.test_run.test.cmd_args
        operation = str(getattr(cmd_args, "metric_operation", "all")).lower()
        metric_name = str(getattr(cmd_args, "metric_name", "bw")).lower()
        aggregate = str(getattr(cmd_args, "metric_aggregate", "sum")).lower()

        return _aggregate_values(_metric_values(_filter_rows(rows, operation), metric_name), aggregate)

    def get_metric(self, metric: str) -> MetricValue:
        rows = extract_fio_data(self.results_file)
        if not rows:
            return METRIC_ERROR

        by_operation = {row.operation: row for row in rows}
        if metric == "default":
            return self._configured_default_metric(rows)
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
