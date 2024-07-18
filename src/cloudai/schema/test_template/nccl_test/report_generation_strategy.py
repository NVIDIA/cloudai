# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import re
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

    def can_handle_directory(self, directory_path: str) -> bool:
        stdout_path = os.path.join(directory_path, "stdout.txt")
        if os.path.exists(stdout_path):
            with open(stdout_path, "r") as file:
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

    def generate_report(self, test_name: str, directory_path: str, sol: Optional[float] = None) -> None:
        report_data, _ = self._parse_output(directory_path)
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
            df["Size (B)"] = df["Size (B)"].astype(float)
            df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
            df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
            df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
            df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)
            df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")
            self._generate_bokeh_report(test_name, df, directory_path, sol)
            self._generate_csv_report(df, directory_path)

    def _parse_output(self, directory_path: str) -> Tuple[List[List[str]], Optional[float]]:
        """
        Extract data from 'stdout.txt' for report generation.

        Args:
            directory_path (str): Directory containing 'stdout.txt'.

        Returns:
            Tuple[List[List[str]], Optional[float]]: Parsed data and optional average bus bandwidth.
        """
        stdout_path = os.path.join(directory_path, "stdout.txt")
        avg_bus_bw = None
        data = []
        if os.path.isfile(stdout_path):
            with open(stdout_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if re.match(r"^\d", line):
                        data.append(re.split(r"\s+", line))
                content = file.read()
                avg_bus_bw_match = re.search(r"Avg bus bandwidth\s+:\s+(\d+\.\d+)", content)
                avg_bus_bw = float(avg_bus_bw_match.group(1)) if avg_bus_bw_match else None
        return data, avg_bus_bw

    def _generate_bokeh_report(
        self, test_name: str, df: pd.DataFrame, directory_path: str, sol: Optional[float]
    ) -> None:
        """
        Create and saves plots to visualize NCCL test metrics.

        Args:
            test_name (str): The name of the test.
            df (pd.DataFrame): DataFrame containing the NCCL test data.
            directory_path (str): Output directory path for saving the plots.
            sol (Optional[float]): Speed-of-light performance for reference.
        """
        report_tool = BokehReportTool(directory_path)
        line_plots = [
            ("Busbw (GB/s) Out-of-place", "blue", "Out-of-place Bus Bandwidth"),
            ("Busbw (GB/s) In-place", "green", "In-place Bus Bandwidth"),
        ]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_single_line_plot(
                title=f"{test_name} {title}",
                x_column="Size (B)",
                y_column=col_name,
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=sol,
                color=color,
            )

        combined_columns = [("Busbw (GB/s) Out-of-place", "blue"), ("Busbw (GB/s) In-place", "green")]
        report_tool.add_log_x_linear_y_multi_line_plot(
            title=f"{test_name} Combined Bus Bandwidth",
            x_column="Size (B)",
            y_columns=combined_columns,
            x_axis_label="Message Size",
            y_axis_label="Bandwidth (GB/s)",
            df=df,
            sol=sol,
        )

        report_tool.finalize_report("cloudai_nccl_test_bokeh_report.html")

    def _generate_csv_report(self, df: pd.DataFrame, directory_path: str) -> None:
        """
        Generate a CSV report from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the NCCL test data.
            directory_path (str): Output directory path for saving the CSV report.
        """
        csv_report_tool = CSVReportTool(directory_path)
        csv_report_tool.set_dataframe(df)
        csv_report_tool.finalize_report("cloudai_nccl_test_csv_report.csv")
