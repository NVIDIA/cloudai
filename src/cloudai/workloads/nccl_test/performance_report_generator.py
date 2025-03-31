# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool
from cloudai.report_generator.util import add_human_readable_sizes


class NcclTestPerformanceReportGenerator:
    """Extract and transform NCCL test output data and generate reports."""

    def __init__(self, output_path: Path, test_name: str, sol: Optional[float] = None):
        self.stdout_path = output_path / "stdout.txt"
        self.output_path = output_path
        self.test_name = test_name
        self.sol = sol

    def generate(self) -> None:
        df, _ = self._extract_data()
        if df.empty:
            return

        self._generate_csv_report(df)
        self._generate_bokeh_report(df)

    def _extract_data(self) -> Tuple[pd.DataFrame, Optional[float]]:
        data, avg_bus_bw = self._parse_stdout()
        if not data:
            return pd.DataFrame(), None

        df = pd.DataFrame(
            data,
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

        df["Size (B)"] = df["Size (B)"].astype(int)
        df["Time (us) Out-of-place"] = df["Time (us) Out-of-place"].astype(float).round(1)
        df["Time (us) In-place"] = df["Time (us) In-place"].astype(float).round(1)
        df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
        df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
        df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
        df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)

        df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")

        gpu_type, num_devices_per_node, num_ranks = self._extract_device_info()

        df["gpu_type"] = gpu_type
        df["num_devices_per_node"] = num_devices_per_node
        df["num_ranks"] = num_ranks

        return df, avg_bus_bw

    def _extract_device_info(self) -> Tuple[str, int, int]:
        gpu_type, num_ranks = "Unknown", 0
        device_indices = {}

        if not self.stdout_path.is_file():
            return gpu_type, 0, 0

        with self.stdout_path.open(encoding="utf-8") as file:
            for line in file:
                if "Rank" in line and "device" in line and "NVIDIA" in line:
                    num_ranks += 1

                    if match := re.search(r"on\s+([\w\d\-.]+)\s+device\s+(\d+)", line):
                        host, device_index = match.groups()
                        device_indices[host] = max(device_indices.get(host, -1), int(device_index))

                    if match := re.search(r"NVIDIA\s+([A-Z0-9]+)", line):
                        gpu_type = match.group(1).strip()

        num_devices = max(device_indices.values(), default=-1) + 1 if device_indices else 0
        return gpu_type, num_devices, num_ranks

    def _parse_stdout(self) -> Tuple[List[List[str]], Optional[float]]:
        data = []
        avg_bus_bw = None

        if not self.stdout_path.is_file():
            return data, avg_bus_bw

        with self.stdout_path.open("r") as file:
            for line in file:
                line = line.strip()
                if re.match(r"^\d", line):
                    data.append(re.split(r"\s+", line))

            content = file.read()
            avg_bus_bw_match = re.search(r"Avg bus bandwidth\s+:\s+(\d+\.\d+)", content)
            avg_bus_bw = float(avg_bus_bw_match.group(1)) if avg_bus_bw_match else None

        return data, avg_bus_bw

    def _generate_csv_report(self, df: pd.DataFrame) -> None:
        csv_report_tool = CSVReportTool(self.output_path)
        csv_report_tool.set_dataframe(df)
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_csv_report.csv"))

    def _generate_bokeh_report(self, df: pd.DataFrame) -> None:
        report_tool = BokehReportTool(self.output_path)

        line_plots = [
            ("Busbw (GB/s) Out-of-place", "blue", "Out-of-place Bus Bandwidth"),
            ("Busbw (GB/s) In-place", "green", "In-place Bus Bandwidth"),
        ]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_multi_line_plot(
                title=f"{self.test_name} {title}",
                x_column="Size (B)",
                y_columns=[(col_name, color)],
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=self.sol,
            )

        combined_columns = [("Busbw (GB/s) Out-of-place", "blue"), ("Busbw (GB/s) In-place", "green")]
        report_tool.add_log_x_linear_y_multi_line_plot(
            title=f"{self.test_name} Combined Bus Bandwidth",
            x_column="Size (B)",
            y_columns=combined_columns,
            x_axis_label="Message Size",
            y_axis_label="Bandwidth (GB/s)",
            df=df,
            sol=self.sol,
        )

        report_tool.finalize_report(Path("cloudai_nccl_test_bokeh_report.html"))