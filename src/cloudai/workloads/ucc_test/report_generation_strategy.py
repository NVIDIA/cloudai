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
import logging
import re
from functools import cache
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.util import add_human_readable_sizes


@cache
def parse_ucc_output(res_file: Path) -> Optional[pd.DataFrame]:
    data = []
    with res_file.open("r") as file:
        content = file.read()
        for line in content.splitlines()[14:]:  # UCC data starts at line 15
            values = re.split(r"\s+", line.strip())
            if len(values) == 8:  # Expected data format
                data.append(values)

    if not data:
        return None

    return pd.DataFrame(
        data,
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


class UCCTestReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from UCC test outputs.

    Visualizing bus bandwidth changes over epochs using interactive Bokeh plots.
    """

    metrics: ClassVar[list[str]] = ["default", "time-avg", "time-max"]

    def can_handle_directory(self) -> bool:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                content = file.read()
                if re.search(r"\b(avg\s+min\s+max\s+avg\s+max\s+min)\b", content):
                    return True
        return False

    def generate_report(self) -> None:
        df = parse_ucc_output(self.test_run.output_path / "stdout.txt")

        if df is None:
            logging.warning(f"Could not extract data from UCC report in {self.test_run.output_path}")
            return

        df["Size (B)"] = df["Size (B)"].astype(float)
        df["Bandwidth (GB/s) avg"] = df["Bandwidth (GB/s) avg"].astype(float)
        df["Bandwidth (GB/s) max"] = df["Bandwidth (GB/s) max"].astype(float)
        df["Bandwidth (GB/s) min"] = df["Bandwidth (GB/s) min"].astype(float)
        df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")
        self._generate_plots(df)

    def _generate_plots(self, df: pd.DataFrame) -> None:
        """
        Create and saves plots to visualize UCC test metrics.

        Args:
            df (pd.DataFrame): DataFrame containing the NCCL test data.
        """
        report_tool = BokehReportTool(self.test_run.output_path)
        line_plots = [("Bandwidth (GB/s) avg", "black", "Average Bandwidth")]
        for col_name, color, title in line_plots:
            report_tool.add_log_x_linear_y_multi_line_plot(
                title=title,
                x_column="Size (B)",
                y_columns=[(col_name, color)],
                x_axis_label="Message Size",
                y_axis_label="Bandwidth (GB/s)",
                df=df,
                sol=self.test_run.sol,
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
            sol=self.test_run.sol,
        )

        report_tool.finalize_report(Path("cloudai_ucc_test_bokeh_report.html"))

    def get_metric(self, metric: str) -> float:
        """
        Calculate the metric value from the UCC test output.

        Today we expect to optimize for a single message size only, user should
        limit the number of message sizes in the test configuration. CloudAI will
        take the first value from the list.
        """
        res_data = parse_ucc_output(self.test_run.output_path / "stdout.txt")
        if metric not in {"default", "time-avg", "time-max"} or res_data is None:
            return METRIC_ERROR

        label_to_metric = {
            "default": "Time Avg (us)",
            "time-avg": "Time Avg (us)",
            "time-max": "Time Max (us)",
        }
        values: list[float] = res_data[label_to_metric[metric]].to_list()
        if len(values) != 1:
            logging.warning(
                "Expected to optimize for a single message size only, but got %d values",
                len(values),
            )
        return values[0]
