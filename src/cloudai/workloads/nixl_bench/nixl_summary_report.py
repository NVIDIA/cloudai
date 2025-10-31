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
from cloudai.util.lazy_imports import lazy

from .nixl_bench import NIXLBenchTestDefinition

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class NIXLBenchComparisonReport(ComparisonReport):
    """Comparison report for NIXL Bench."""

    INFO_COLUMNS = ("block_size", "batch_size")

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nixl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test.test_definition, NIXLBenchTestDefinition)]

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
                        data_columns=["avg_lat"],
                    ),
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["bw_gb_sec"],
                    ),
                ]
            )
        return tables

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            charts.extend(
                [
                    self.create_chart(group, dfs, "Latency", list(self.INFO_COLUMNS), ["avg_lat"], "Time (us)"),
                    self.create_chart(group, dfs, "Bandwidth", list(self.INFO_COLUMNS), ["bw_gb_sec"], "Busbw (GB/s)"),
                ]
            )
        return charts

    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        if (tr.output_path / "nixlbench.csv").exists():
            return lazy.pd.read_csv(tr.output_path / "nixlbench.csv")
        return lazy.pd.DataFrame(
            {
                "block_size": lazy.pd.Series([], dtype=int),
                "batch_size": lazy.pd.Series([], dtype=int),
                "avg_lat": lazy.pd.Series([], dtype=float),
                "bw_gb_sec": lazy.pd.Series([], dtype=float),
            }
        )
