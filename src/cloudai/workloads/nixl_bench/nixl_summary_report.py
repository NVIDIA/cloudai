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

from pathlib import Path
from typing import TYPE_CHECKING

import cloudai.metrics
from cloudai.core import System, TestRun, TestScenario
from cloudai.report_generator.comparison_report import (
    ComparisonReport,
    ComparisonReportConfig,
    ComparisonSection,
    MetricColumn,
)
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.report_generator.util import add_human_readable_sizes
from cloudai.util.lazy_imports import lazy

from .nixl_bench import NIXLBenchTestDefinition

if TYPE_CHECKING:
    import pandas as pd


class NIXLBenchComparisonReport(ComparisonReport):
    """Comparison report for NIXL Bench."""

    INFO_COLUMNS = ("block_size", "batch_size")
    BLOCK_SIZE_LABEL_COLUMN = "block_size_human_readable"

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nixl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, NIXLBenchTestDefinition)]

    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
        sections: list[ComparisonSection] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            sections.extend(
                [
                    ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Latency",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["avg_lat"],
                        y_axis_label="Time (us)",
                        x_axis_type="indexed_category",
                        x_axis_column=self.BLOCK_SIZE_LABEL_COLUMN,
                        x_axis_label="Payload size",
                        metric_columns={
                            "avg_lat": MetricColumn(
                                cloudai.metrics.TRANSFER_LATENCY,
                                coordinate_columns={
                                    "payload_size_bytes": "block_size",
                                    "batch_size": "batch_size",
                                },
                            )
                        },
                    ),
                    ComparisonSection(
                        group=group,
                        dfs=dfs,
                        title="Bandwidth",
                        info_columns=list(self.INFO_COLUMNS),
                        data_columns=["bw_gb_sec"],
                        y_axis_label="Busbw (GB/s)",
                        x_axis_type="indexed_category",
                        x_axis_column=self.BLOCK_SIZE_LABEL_COLUMN,
                        x_axis_label="Payload size",
                        metric_columns={
                            "bw_gb_sec": MetricColumn(
                                cloudai.metrics.TRANSFER_BANDWIDTH,
                                coordinate_columns={
                                    "payload_size_bytes": "block_size",
                                    "batch_size": "batch_size",
                                },
                            )
                        },
                    ),
                ]
            )
        return sections

    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        if (tr.output_path / "nixlbench.csv").exists():
            df = lazy.pd.read_csv(tr.output_path / "nixlbench.csv")
            return add_human_readable_sizes(df, "block_size", self.BLOCK_SIZE_LABEL_COLUMN)
        return lazy.pd.DataFrame(
            {
                "block_size": lazy.pd.Series([], dtype=int),
                self.BLOCK_SIZE_LABEL_COLUMN: lazy.pd.Series([], dtype=str),
                "batch_size": lazy.pd.Series([], dtype=int),
                "avg_lat": lazy.pd.Series([], dtype=float),
                "bw_gb_sec": lazy.pd.Series([], dtype=float),
            }
        )
