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

import re
from functools import cache
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Tuple

from cloudai.core import METRIC_ERROR, System, TestRun
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool
from cloudai.report_generator.util import add_human_readable_sizes
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd

from .report_generation_strategy import NcclTestReportGenerationStrategy


def _parse_data_rows(file: TextIOWrapper) -> List[List[str]]:
    parsed_data_rows: List[List[str]] = []
    for line in file:
        line: str = line.strip()
        if re.match(r"^\d", line):
            parts = re.split(r"\s+", line)
            if len(parts) == 13:
                parsed_data_rows.append(parts)
    return parsed_data_rows


def _parse_device_info(file: TextIOWrapper) -> Tuple[int, str, int]:
    device_indices: dict = {}
    gpu_type: str = "Unknown"
    num_ranks: int = 0

    for line in file:
        if "Rank" in line and "device" in line and "NVIDIA" in line:
            num_ranks += 1

            if match := re.search(r"on\s+([\w\d\-.]+)\s+device\s+(\d+)", line):
                host, device_index = match.groups()
                device_indices[host] = max(device_indices.get(host, -1), int(device_index))

            if match := re.search(r"NVIDIA\s+(.+?)(?=\s*$|\s+\[|\s+device)", line):
                gpu_type = match.group(1).strip()

    num_devices_per_node: int = max(device_indices.values(), default=-1) + 1 if device_indices else 0
    return num_devices_per_node, gpu_type, num_ranks


@cache
def extract_nccl_data(stdout_path: Path) -> Tuple[List[List[str]], str, int, int]:
    parsed_data_rows: List[List[str]] = []
    gpu_type: str = "Unknown"
    num_ranks: int = 0
    num_devices_per_node: int = 0

    if not stdout_path.is_file():
        return parsed_data_rows, gpu_type, num_devices_per_node, num_ranks

    with stdout_path.open("r", encoding="utf-8") as file:
        parsed_data_rows = _parse_data_rows(file)
        file.seek(0)
        num_devices_per_node, gpu_type, num_ranks = _parse_device_info(file)

    return parsed_data_rows, gpu_type, num_devices_per_node, num_ranks


class NcclTestPerformanceReportGenerationStrategy(NcclTestReportGenerationStrategy):
    """Strategy for generating performance reports from NCCL test outputs."""

    metrics: ClassVar[list[str]] = ["default", "latency-in-place", "latency-out-of-place"]

    def __init__(self, system: System, tr: TestRun) -> None:
        super().__init__(system, tr)

    def generate_report(self) -> None:
        df: pd.DataFrame = self._extract_data()
        if df.empty:
            return

        self._generate_csv_report(df)
        self._generate_bokeh_report(df)

    def _parse_stdout(self) -> Tuple[List[List[str]], str, int, int]:
        return extract_nccl_data(self.test_run.output_path / "stdout.txt")

    def _extract_data(self) -> pd.DataFrame:
        parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = self._parse_stdout()
        if not parsed_data_rows:
            return lazy.pd.DataFrame()

        df: pd.DataFrame = lazy.pd.DataFrame(
            parsed_data_rows,
            columns=[
                "Size (B)",
                "Count",
                "Type",
                "Redop",
                "Root",
                "Time (us) Out-of-place",
                "Algbw (GB/s) Out-of-place",
                "Busbw (GB/s) Out-of-place",
                "#Wrong Out-of-place",
                "Time (us) In-place",
                "Algbw (GB/s) In-place",
                "Busbw (GB/s) In-place",
                "#Wrong In-place",
            ],
        )

        df["GPU Type"] = gpu_type
        df["Devices per Node"] = num_devices_per_node
        df["Ranks"] = num_ranks

        df["Size (B)"] = df["Size (B)"].astype(int)
        df["Time (us) Out-of-place"] = df["Time (us) Out-of-place"].astype(float).round(2)
        df["Time (us) In-place"] = df["Time (us) In-place"].astype(float).round(2)
        df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
        df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
        df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
        df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)

        df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")

        return df

    def _generate_csv_report(self, df: pd.DataFrame) -> None:
        csv_report_tool: CSVReportTool = CSVReportTool(self.test_run.output_path)
        csv_report_tool.set_dataframe(df)
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_csv_report.csv"))

    def _generate_bokeh_report(self, df: pd.DataFrame) -> None:
        report_tool: BokehReportTool = BokehReportTool(self.test_run.output_path)

        line_plots: List[Tuple[str, str, str]] = [
            ("Busbw (GB/s) Out-of-place", "blue", "Out-of-place Bus Bandwidth"),
            ("Busbw (GB/s) In-place", "green", "In-place Bus Bandwidth"),
        ]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_multi_line_plot(
                title=f"{self.test_run.name} {title}",
                x_column="Size (B)",
                y_columns=[(col_name, color)],
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=self.test_run.sol,
            )

        combined_columns: List[Tuple[str, str]] = [
            ("Busbw (GB/s) Out-of-place", "blue"),
            ("Busbw (GB/s) In-place", "green"),
        ]
        report_tool.add_log_x_linear_y_multi_line_plot(
            title=f"{self.test_run.name} Combined Bus Bandwidth",
            x_column="Size (B)",
            y_columns=combined_columns,
            x_axis_label="Message Size",
            y_axis_label="Bandwidth (GB/s)",
            df=df,
            sol=self.test_run.sol,
        )

        report_tool.finalize_report(Path("cloudai_nccl_test_bokeh_report.html"))

    def get_metric(self, metric: str) -> float:
        df: pd.DataFrame = self._extract_data()
        if df.empty:
            return METRIC_ERROR

        metric_to_field = {
            "default": "Time (us) In-place",
            "latency-in-place": "Time (us) In-place",
            "latency-out-of-place": "Time (us) Out-of-place",
        }

        if metric not in metric_to_field:
            return METRIC_ERROR

        data = df[metric_to_field[metric]].astype(float).tolist()

        return float(lazy.np.mean(data))
