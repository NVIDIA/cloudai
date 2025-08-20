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

from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from cloudai.core import System, TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonReportConfig
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.report_generator.util import (
    add_human_readable_sizes,
)
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition

from .performance_report_generation_strategy import extract_nccl_data

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class NcclComparisonReport(ComparisonReport):
    """Comparison report for NCCL Test."""

    INFO_COLUMNS = ("Size (B)", "Count", "Type", "Redop")
    LATENCY_DATA_COLUMNS = ("Time (us) Out-of-place", "Time (us) In-place")
    BANDWIDTH_DATA_COLUMNS = ("Busbw (GB/s) Out-of-place", "Busbw (GB/s) In-place")

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nccl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test.test_definition, NCCLTestDefinition)]

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        tables: list[Table] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            tables.extend(
                [
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Latency",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=list(self.LATENCY_DATA_COLUMNS),
                    ),
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=list(self.BANDWIDTH_DATA_COLUMNS),
                    ),
                ]
            )
        return tables

    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = extract_nccl_data(tr.output_path / "stdout.txt")
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

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            if chart := self.create_chart(
                group, dfs, "Latecy", list(self.INFO_COLUMNS), list(self.LATENCY_DATA_COLUMNS), "Time (us)"
            ):
                charts.append(chart)
            if chart := self.create_chart(
                group, dfs, "Bandwidth", list(self.INFO_COLUMNS), list(self.BANDWIDTH_DATA_COLUMNS), "Busbw (GB/s)"
            ):
                charts.append(chart)
        return charts
