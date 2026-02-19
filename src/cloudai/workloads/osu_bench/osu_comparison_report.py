# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import TYPE_CHECKING

from rich.table import Table

from cloudai.core import System, TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonReportConfig
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.util.lazy_imports import lazy

from .osu_bench import OSUBenchTestDefinition

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class OSUBenchComparisonReport(ComparisonReport):
    """Comparison report for OSU Bench."""

    INFO_COLUMNS = ("size",)

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "osu_bench_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, OSUBenchTestDefinition)]

    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        csv_path = tr.output_path / "osu_bench.csv"
        if not csv_path.exists():
            return lazy.pd.DataFrame()

        df = lazy.pd.read_csv(csv_path)

        if "size" not in df.columns:
            logging.warning("%s: missing 'size' column, skipping", csv_path)
            return lazy.pd.DataFrame()

        df["size"] = df["size"].astype(int)
        return df

    @staticmethod
    def _has_metric(dfs: list["pd.DataFrame"], col: str) -> bool:
        """Only include a metric if all compared DataFrames have it."""
        return bool(dfs) and all((col in df.columns) and df[col].notna().any() for df in dfs)

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        tables: list[Table] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]

            if self._has_metric(dfs, "avg_lat"):
                tables.append(
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Latency",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["avg_lat"],
                    )
                )
            if self._has_metric(dfs, "mb_sec"):
                tables.append(
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["mb_sec"],
                    )
                )
            if self._has_metric(dfs, "messages_sec"):
                tables.append(
                    self.create_table(
                        group,
                        dfs=dfs,
                        title="Message Rate",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["messages_sec"],
                    )
                )

        return tables

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]

            if self._has_metric(dfs, "avg_lat"):
                charts.append(
                    self.create_chart(
                        group,
                        dfs,
                        "Latency",
                        list(self.INFO_COLUMNS),
                        ["avg_lat"],
                        "Time (us)",
                    )
                )
            if self._has_metric(dfs, "mb_sec"):
                charts.append(
                    self.create_chart(
                        group,
                        dfs,
                        "Bandwidth",
                        list(self.INFO_COLUMNS),
                        ["mb_sec"],
                        "Bandwidth (MB/s)",
                    )
                )
            if self._has_metric(dfs, "messages_sec"):
                charts.append(
                    self.create_chart(
                        group,
                        dfs,
                        "Message Rate",
                        list(self.INFO_COLUMNS),
                        ["messages_sec"],
                        "Messages/s",
                    )
                )

        return charts
