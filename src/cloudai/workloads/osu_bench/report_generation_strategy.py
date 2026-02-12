# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from enum import Enum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from cloudai.core import ReportGenerationStrategy
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class BenchmarkType(Enum):
    """Type of benchmark to extract data from."""

    BANDWIDTH = 0
    """Bandwidth benchmark."""

    MULTIPLE_BANDWIDTH = 1
    """Multiple bandwidth benchmark."""

    LATENCY = 2
    """Latency benchmark."""


HEADERS = {
    BenchmarkType.LATENCY: (
        r"#\s*Size\s+Avg Latency\(us\)"
        r"(?:\s+Min Latency\(us\)\s+Max Latency\(us\)\s+Iterations)?"
    ),
    BenchmarkType.MULTIPLE_BANDWIDTH: r"#\s*Size\s+MB/s\s+Messages/s",
    BenchmarkType.BANDWIDTH: r"#\s*Size\s+Bandwidth\s*\(MB/s\)",
}


def _detect_benchmark_type(line: str) -> BenchmarkType | None:
    for b_type, header in HEADERS.items():
        if re.match(header, line):
            return b_type

    return None


def _parse_data_row(parts: list[str], benchmark_type: BenchmarkType) -> list[str] | None:
    if len(parts) < 2:
        return None

    try:
        int(parts[0])  # message size
    except ValueError:
        return None

    # Append row data based on benchmark type.
    if benchmark_type == BenchmarkType.MULTIPLE_BANDWIDTH:
        if len(parts) >= 3:
            # size, MB/s, Messages/s
            return [parts[0], parts[1], parts[2]]
        return None

    if benchmark_type == BenchmarkType.BANDWIDTH:
        # size, MB/s
        return [parts[0], parts[1]]

    # LATENCY
    return [parts[0], parts[1]]


def _columns_for_type(benchmark_type: BenchmarkType) -> list[str]:
    if benchmark_type == BenchmarkType.MULTIPLE_BANDWIDTH:
        return ["size", "mb_sec", "messages_sec"]

    if benchmark_type == BenchmarkType.BANDWIDTH:
        return ["size", "mb_sec"]

    return ["size", "avg_lat"]


@cache
def extract_osu_bench_data(stdout_file: Path) -> pd.DataFrame:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return lazy.pd.DataFrame()

    data: list[list[str]] = []
    benchmark_type: BenchmarkType | None = None

    for line in stdout_file.read_text().splitlines():
        if benchmark_type is None:
            benchmark_type = _detect_benchmark_type(line)
            continue

        if row := _parse_data_row(line.split(), benchmark_type):
            data.append(row)

    if benchmark_type is None:
        return lazy.pd.DataFrame()

    columns = _columns_for_type(benchmark_type)
    df = lazy.pd.DataFrame(data, columns=lazy.pd.Index(columns))

    df["size"] = df["size"].astype(int)

    if "mb_sec" in df.columns:
        df["mb_sec"] = df["mb_sec"].astype(float)

    if "messages_sec" in df.columns:
        df["messages_sec"] = df["messages_sec"].astype(float)

    if "avg_lat" in df.columns:
        df["avg_lat"] = df["avg_lat"].astype(float)

    return df


class OSUBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for OSU Bench."""

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        df = extract_osu_bench_data(self.results_file)
        return not df.empty

    def generate_report(self) -> None:
        if not self.can_handle_directory():
            return

        df = extract_osu_bench_data(self.results_file)
        df.to_csv(self.test_run.output_path / "osu_bench.csv", index=False)
