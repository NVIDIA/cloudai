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

from cloudai import ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool
from cloudai.report_generator.util import add_human_readable_sizes


class NcclTestReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from NCCL test outputs.

    Visualizing bus bandwidth changes over epochs using interactive Bokeh plots.
    """

    def can_handle_directory(self) -> bool:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                content = file.read()
                if re.search(r"out-of-place|in-place", content) and re.search(
                    r"\b(size\s+count\s+type\s+redop\s+root\s+"
                    r"time\s+algbw\s+busbw\s+#wrong\s+time\s+"
                    r"algbw\s+busbw\s+#wrong)\b",
                    content,
                    re.IGNORECASE,
                ):
                    return True
        return False

    def generate_report(self) -> None:
        report_data, _ = self._parse_output()
        if report_data:
            df = pd.DataFrame(
                report_data,
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

            self._generate_bokeh_report(self.test_run.name, df)
            self._generate_csv_report(df)

    def _parse_output(self) -> Tuple[List[List[str]], Optional[float]]:
        """
        Extract data from 'stdout.txt' for report generation.

        Returns:
            Tuple[List[List[str]], Optional[float]]: Parsed data and optional average bus bandwidth.
        """
        stdout_path = self.test_run.output_path / "stdout.txt"
        avg_bus_bw = None
        data = []
        if stdout_path.is_file():
            with stdout_path.open("r") as file:
                for line in file:
                    line = line.strip()
                    if re.match(r"^\d", line):
                        data.append(re.split(r"\s+", line))
                content = file.read()
                avg_bus_bw_match = re.search(r"Avg bus bandwidth\s+:\s+(\d+\.\d+)", content)
                avg_bus_bw = float(avg_bus_bw_match.group(1)) if avg_bus_bw_match else None
        return data, avg_bus_bw

    def _generate_bokeh_report(self, test_name: str, df: pd.DataFrame) -> None:
        """
        Create and saves plots to visualize NCCL test metrics.

        Args:
            test_name (str): The name of the test.
            df (pd.DataFrame): DataFrame containing the NCCL test data.
        """
        report_tool = BokehReportTool(self.test_run.output_path)
        line_plots = [
            ("Busbw (GB/s) Out-of-place", "blue", "Out-of-place Bus Bandwidth"),
            ("Busbw (GB/s) In-place", "green", "In-place Bus Bandwidth"),
        ]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_multi_line_plot(
                title=f"{test_name} {title}",
                x_column="Size (B)",
                y_columns=[(col_name, color)],
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=self.test_run.sol,
            )

        combined_columns = [("Busbw (GB/s) Out-of-place", "blue"), ("Busbw (GB/s) In-place", "green")]
        report_tool.add_log_x_linear_y_multi_line_plot(
            title=f"{test_name} Combined Bus Bandwidth",
            x_column="Size (B)",
            y_columns=combined_columns,
            x_axis_label="Message Size",
            y_axis_label="Bandwidth (GB/s)",
            df=df,
            sol=self.test_run.sol,
        )

        report_tool.finalize_report(Path("cloudai_nccl_test_bokeh_report.html"))

    def _generate_csv_report(self, df: pd.DataFrame) -> None:
        """
        Generate a CSV report from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the NCCL test data.
        """
        csv_report_tool = CSVReportTool(self.test_run.output_path)
        csv_report_tool.set_dataframe(df)
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_csv_report.csv"))
