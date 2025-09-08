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

import logging
from pathlib import Path
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.common.nixl import extract_nixlbench_data


class NIXLBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NIXL Bench directories."""

    metrics: ClassVar[list[str]] = ["default", "latency"]

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        df = extract_nixlbench_data(self.results_file)
        return not df.empty

    def generate_report(self) -> None:
        if not self.can_handle_directory():
            return

        self.generate_bokeh_report()
        df = extract_nixlbench_data(self.results_file)
        df.to_csv(self.test_run.output_path / "nixlbench.csv", index=False)

    def get_metric(self, metric: str) -> float:
        logging.debug(f"Getting metric {metric} from {self.results_file.absolute()}")
        df = extract_nixlbench_data(self.results_file)
        if df.empty or metric not in {"default", "latency"}:
            return METRIC_ERROR

        return float(lazy.np.mean(df["avg_lat"]))

    def generate_bokeh_report(self) -> None:
        df = extract_nixlbench_data(self.results_file)

        report_tool = BokehReportTool(self.test_run.output_path)
        p = report_tool.add_log_x_linear_y_multi_line_plot(
            title="NIXL Bench Latency",
            df=df,
            x_column="block_size",
            y_columns=[("avg_lat", "blue")],
            x_axis_label="Block Size (B)",
            y_axis_label="Latency (us)",
        )
        p.width, p.height = 800, 500
        p = report_tool.add_log_x_linear_y_multi_line_plot(
            title="NIXL Bench Bandwidth",
            df=df,
            x_column="block_size",
            y_columns=[("bw_gb_sec", "blue")],
            x_axis_label="Block Size (B)",
            y_axis_label="Bandwidth (GB/Sec)",
        )
        p.width, p.height = 800, 500
        report_tool.finalize_report(Path("cloudai_nixlbench_bokeh_report.html"))
