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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.util.lazy_imports import lazy

from .nixl_bench import aggregate_nixlbench_results, read_nixlbench_results

if TYPE_CHECKING:
    import pandas as pd


class NIXLBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NIXL Bench directories."""

    metrics: ClassVar[list[str]] = ["default", "latency"]

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        return not self._read_results().empty

    def _read_results(self) -> pd.DataFrame:
        try:
            return read_nixlbench_results(self.test_run.output_path)
        except ValueError as exc:
            logging.warning(str(exc))
            return lazy.pd.DataFrame()

    def generate_report(self) -> None:
        df = self._read_results()
        if df.empty:
            return

        self.generate_bokeh_report()
        if "task_id" in df.columns:
            df.to_csv(self.test_run.output_path / "nixlbench_per_task.csv", index=False)
        aggregate_nixlbench_results(df).to_csv(self.test_run.output_path / "nixlbench.csv", index=False)

    def get_metric(self, metric: str) -> MetricValue:
        logging.debug(f"Getting metric {metric} from {self.results_file.absolute()}")
        df = aggregate_nixlbench_results(self._read_results())
        if df.empty or metric not in {"default", "latency"}:
            return METRIC_ERROR

        return float(lazy.np.mean(df["avg_lat"]))

    def generate_bokeh_report(self) -> None:
        df = aggregate_nixlbench_results(self._read_results())
        if df.empty:
            return
        independent_tasks = "task_count" in df.columns

        report_tool = BokehReportTool(self.test_run.output_path)
        bandwidth_columns = [("bw_gb_sec", "blue")]
        if independent_tasks:
            df = df.rename(
                columns={
                    "bw_gb_sec": "Mean per-task bandwidth",
                    "bw_min_gb_sec": "Minimum per-task bandwidth",
                    "bw_sum_gb_sec": "Sum of task bandwidths",
                }
            )
            bandwidth_columns = [
                ("Mean per-task bandwidth", "blue"),
                ("Minimum per-task bandwidth", "orange"),
                ("Sum of task bandwidths", "green"),
            ]
        groups = df.groupby("batch_size") if independent_tasks else [(None, df)]
        for batch_size, frame in groups:
            suffix = f" — Batch size {batch_size}" if independent_tasks else ""
            latency_title = "NIXL Bench Mean Per-Task Latency" if independent_tasks else "NIXL Bench Latency"
            p = report_tool.add_log_x_linear_y_multi_line_plot(
                title=latency_title + suffix,
                df=frame,
                x_column="block_size",
                y_columns=[("avg_lat", "blue")],
                x_axis_label="Block Size (B)",
                y_axis_label="Latency (us)",
            )
            p.width, p.height = 800, 500
            p = report_tool.add_log_x_linear_y_multi_line_plot(
                title="NIXL Bench Bandwidth" + suffix,
                df=frame,
                x_column="block_size",
                y_columns=bandwidth_columns,
                x_axis_label="Block Size (B)",
                y_axis_label="Bandwidth (GB/Sec)",
            )
            p.width, p.height = 800, 500
        report_tool.finalize_report(Path("cloudai_nixlbench_bokeh_report.html"))
