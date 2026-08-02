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

import pathlib
from typing import TYPE_CHECKING

import cloudai.core
import cloudai.report_generator.comparison_report
import cloudai.report_generator.groups
import cloudai.workloads.nccl_test.nccl as nccl
import cloudai.workloads.nccl_test.performance_report_generation_strategy as performance_report_generation_strategy
from cloudai.report_generator.comparison_report import ComparisonReportConfig as ComparisonReportConfig
from cloudai.report_generator.util import (
    add_human_readable_sizes,
)
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class NcclComparisonReport(cloudai.report_generator.comparison_report.ComparisonReport):
    """Comparison report for NCCL Test."""

    INFO_COLUMNS = ("Size (B)", "Count", "Type", "Redop")
    LATENCY_DATA_COLUMNS = ("Time (us) Out-of-place", "Time (us) In-place")
    BANDWIDTH_DATA_COLUMNS = ("Busbw (GB/s) Out-of-place", "Busbw (GB/s) In-place")

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nccl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, nccl.NCCLTestDefinition)]

    def build_sections(
        self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]
    ) -> list[cloudai.report_generator.comparison_report.ComparisonSection]:
        sections: list[cloudai.report_generator.comparison_report.ComparisonSection] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            sections.extend(
                [
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Latency",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=list(self.LATENCY_DATA_COLUMNS),
                        y_axis_label="Time (us)",
                        x_axis_type="indexed_category",
                        x_axis_column="Size Human-readable",
                        x_axis_label="Message size",
                        legacy_chart_title="Latecy",
                    ),
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=list(self.BANDWIDTH_DATA_COLUMNS),
                        y_axis_label="Busbw (GB/s)",
                        x_axis_type="indexed_category",
                        x_axis_column="Size Human-readable",
                        x_axis_label="Message size",
                    ),
                ]
            )
        return sections

    def extract_data_as_df(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
        parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = (
            performance_report_generation_strategy.extract_nccl_data(tr.output_path / "stdout.txt")
        )
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
