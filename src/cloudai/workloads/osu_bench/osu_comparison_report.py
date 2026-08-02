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
import pathlib
from typing import TYPE_CHECKING

import cloudai.core
import cloudai.report_generator.comparison_report
import cloudai.report_generator.groups
import cloudai.workloads.osu_bench.osu_bench as osu_bench
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class OSUBenchComparisonReport(cloudai.report_generator.comparison_report.ComparisonReport):
    """Comparison report for OSU Bench."""

    INFO_COLUMNS = ("size",)

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "osu_bench_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, osu_bench.OSUBenchTestDefinition)]

    def extract_data_as_df(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
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

    def build_sections(
        self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]
    ) -> list[cloudai.report_generator.comparison_report.ComparisonSection]:
        sections: list[cloudai.report_generator.comparison_report.ComparisonSection] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]

            if self._has_metric(dfs, "avg_lat"):
                sections.append(
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Latency",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["avg_lat"],
                        y_axis_label="Time (us)",
                    )
                )
            if self._has_metric(dfs, "mb_sec"):
                sections.append(
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["mb_sec"],
                        y_axis_label="Bandwidth (MB/s)",
                    )
                )
            if self._has_metric(dfs, "messages_sec"):
                sections.append(
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Message Rate",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["messages_sec"],
                        y_axis_label="Messages/s",
                    )
                )

        return sections
