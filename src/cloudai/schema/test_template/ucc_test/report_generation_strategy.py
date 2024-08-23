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
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from cloudai import ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.util import add_human_readable_sizes


class UCCTestReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from UCC test outputs.

    Visualizing bus bandwidth changes over epochs using interactive Bokeh plots.
    """

    def can_handle_directory(self, directory_path: Path) -> bool:
        stdout_path = directory_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                content = file.read()
                if re.search(r"\b(avg\s+min\s+max\s+avg\s+max\s+min)\b", content):
                    return True
        return False

    def generate_report(self, test_name: str, directory_path: Path, sol: Optional[float] = None) -> None:
        report_data = []
        stdout_path = directory_path / "stdout.txt"
        if stdout_path.is_file():
            with stdout_path.open("r") as file:
                content = file.read()
                report_data = self._parse_output(content)

        if report_data:
            df = pd.DataFrame(
                report_data,
                columns=[
                    "Count",
                    "Size (B)",
                    "Time Avg (us)",
                    "Time Min (us)",
                    "Time Max (us)",
                    "Bandwidth (GB/s) avg",
                    "Bandwidth (GB/s) max",
                    "Bandwidth (GB/s) min",
                ],
            )
            df["Size (B)"] = df["Size (B)"].astype(float)
            df["Bandwidth (GB/s) avg"] = df["Bandwidth (GB/s) avg"].astype(float)
            df["Bandwidth (GB/s) max"] = df["Bandwidth (GB/s) max"].astype(float)
            df["Bandwidth (GB/s) min"] = df["Bandwidth (GB/s) min"].astype(float)
            df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")
            self._generate_plots(df, directory_path, sol)

    def _parse_output(self, content: str) -> List[List[str]]:
        """
        Extract data from 'stdout.txt' for report generation.

        Args:
            content (str): Content of the 'stdout.txt' file.
        """
        data = []
        for line in content.splitlines()[14:]:  # UCC data starts at line 15
            values = re.split(r"\s+", line.strip())
            if len(values) == 8:  # Expected data format
                data.append(values)
        return data

    def _generate_plots(self, df: pd.DataFrame, directory_path: Path, sol: Optional[float]) -> None:
        """
        Create and saves plots to visualize UCC test metrics.

        Args:
            df (pd.DataFrame): DataFrame containing the NCCL test data.
            directory_path (Path): Output directory path for saving the plots.
            sol (Optional[float]): Speed-of-light performance for reference.
        """
        report_tool = BokehReportTool(directory_path)
        line_plots = [("Bandwidth (GB/s) avg", "black", "Average Bandwidth")]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_multi_line_plot(
                title=title,
                x_column="Size (B)",
                y_columns=[(col_name, color)],
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=sol,
            )

        combined_columns = [
            ("Bandwidth (GB/s) avg", "blue"),
            ("Bandwidth (GB/s) min", "green"),
            ("Bandwidth (GB/s) max", "black"),
        ]
        report_tool.add_log_x_linear_y_multi_line_plot(
            title="Combined Bus Bandwidth",
            x_column="Size (B)",
            y_columns=combined_columns,
            x_axis_label="Message Size",
            y_axis_label="Bandwidth (GB/s)",
            df=df,
            sol=sol,
        )

        report_tool.finalize_report(Path("cloudai_ucc_test_bokeh_report.html"))
