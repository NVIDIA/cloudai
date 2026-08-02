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
import cloudai.workloads.nixl_bench.nixl_bench as nixl_bench
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class NIXLBenchComparisonReport(cloudai.report_generator.comparison_report.ComparisonReport):
    """Comparison report for NIXL Bench."""

    INFO_COLUMNS = ("block_size", "batch_size")

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nixl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, nixl_bench.NIXLBenchTestDefinition)]

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
                        data_columns=["avg_lat"],
                        y_axis_label="Time (us)",
                    ),
                    cloudai.report_generator.comparison_report.ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["bw_gb_sec"],
                        y_axis_label="Busbw (GB/s)",
                    ),
                ]
            )
        return sections

    def extract_data_as_df(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
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
