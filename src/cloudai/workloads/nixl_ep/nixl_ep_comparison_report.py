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

import itertools
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, cast

import rich.table

import cloudai.core
import cloudai.report_generator.comparison_report
import cloudai.report_generator.groups
import cloudai.workloads.nixl_ep.log_parsing as log_parsing
import cloudai.workloads.nixl_ep.nixl_ep as nixl_ep
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class NixlEPComparisonReport(cloudai.report_generator.comparison_report.ComparisonReport):
    """Comparison report for NIXL EP benchmark runs."""

    NODE_COLUMN: ClassVar[str] = "Node"
    DISPATCH_COMBINE_BW_COLUMN: ClassVar[str] = "Dispatch+Combine BW (GB/s)"
    AVG_TIME_COLUMN: ClassVar[str] = "Avg Time (us)"
    MIN_TIME_COLUMN: ClassVar[str] = "Min Time (us)"
    MAX_TIME_COLUMN: ClassVar[str] = "Max Time (us)"
    DISPATCH_BW_COLUMN: ClassVar[str] = "Dispatch BW (GB/s)"
    COMBINE_BW_COLUMN: ClassVar[str] = "Combine BW (GB/s)"

    METRIC_COLUMNS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("dispatch_combine_bandwidth_gbps", DISPATCH_COMBINE_BW_COLUMN),
        ("avg_time_us", AVG_TIME_COLUMN),
        ("min_time_us", MIN_TIME_COLUMN),
        ("max_time_us", MAX_TIME_COLUMN),
        ("dispatch_bandwidth_gbps", DISPATCH_BW_COLUMN),
        ("combine_bandwidth_gbps", COMBINE_BW_COLUMN),
    )
    BANDWIDTH_COLUMNS: ClassVar[tuple[str, ...]] = (
        DISPATCH_COMBINE_BW_COLUMN,
        DISPATCH_BW_COLUMN,
        COMBINE_BW_COLUMN,
    )
    TIME_COLUMNS: ClassVar[tuple[str, ...]] = (AVG_TIME_COLUMN, MIN_TIME_COLUMN, MAX_TIME_COLUMN)

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nixl_ep_comparison.html"

    def load_test_runs(self) -> None:
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, nixl_ep.NixlEPTestDefinition)]

    def comparison_values(self, tr: cloudai.core.TestRun) -> dict[str, object]:
        return {"case": tr.name.removeprefix("NIXL.EP."), "NUM_NODES": tr.nnodes}

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _metric_value(value: float | None) -> float | str:
        if value is None:
            return "n/a"
        return round(value, 4)

    @classmethod
    def _empty_df(cls) -> pd.DataFrame:
        return lazy.pd.DataFrame(
            {
                cls.NODE_COLUMN: lazy.pd.Series([], dtype=int),
                **{column: lazy.pd.Series([], dtype=object) for _, column in cls.METRIC_COLUMNS},
            }
        )

    def extract_data_as_df(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for node_idx in range(tr.nnodes):
            samples = log_parsing.parse_nixl_ep_bandwidth_samples(tr.output_path / f"nixl-ep-node-{node_idx}.log")
            row: dict[str, object] = {self.NODE_COLUMN: node_idx}
            for sample_attr, column in self.METRIC_COLUMNS:
                values = [value for sample in samples if (value := getattr(sample, sample_attr)) is not None]
                row[column] = self._metric_value(self._mean(values))
            rows.append(row)

        if not rows:
            return self._empty_df()

        return lazy.pd.DataFrame(rows)

    @staticmethod
    def _has_metric(dfs: list[pd.DataFrame], column: str) -> bool:
        for df in dfs:
            if column not in df.columns:
                continue
            if any(isinstance(value, (int, float)) for value in df[column].tolist()):
                return True
        return False

    def _available_columns(self, dfs: list[pd.DataFrame], columns: tuple[str, ...]) -> list[str]:
        return [column for column in columns if self._has_metric(dfs, column)]

    def create_tables(
        self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]
    ) -> list[rich.table.Table]:
        tables: list[rich.table.Table] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            bandwidth_columns = self._available_columns(dfs, self.BANDWIDTH_COLUMNS)
            time_columns = self._available_columns(dfs, self.TIME_COLUMNS)

            for bandwidth_column in bandwidth_columns:
                tables.append(
                    self.create_table(
                        group,
                        dfs=dfs,
                        title=bandwidth_column,
                        info_columns=[self.NODE_COLUMN],
                        data_columns=[bandwidth_column],
                    )
                )
            for time_column in time_columns:
                tables.append(
                    self.create_table(
                        group,
                        dfs=dfs,
                        title=time_column,
                        info_columns=[self.NODE_COLUMN],
                        data_columns=[time_column],
                    )
                )

        return tables

    def _create_metric_bar_chart(
        self,
        group: cloudai.report_generator.groups.GroupedTestRuns,
        dfs: list[pd.DataFrame],
        metric_column: str,
        y_axis_label: str,
    ) -> bk.figure:
        factors: list[tuple[str, str]] = []
        values: list[float] = []
        nodes: list[str] = []
        runs: list[str] = []
        colors: list[str] = []
        color_cycler = itertools.cycle(["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"])
        color_by_run = {item.name: next(color_cycler) for item in group.items}

        for df, item in zip(dfs, group.items, strict=True):
            if metric_column not in df.columns:
                continue
            for _, row in df.iterrows():
                value = row[metric_column]
                if not isinstance(value, (int, float)):
                    continue
                node = f"Node {row[self.NODE_COLUMN]}"
                factors.append((node, item.name))
                values.append(float(value))
                nodes.append(node)
                runs.append(item.name)
                colors.append(color_by_run[item.name])

        x_range = lazy.bokeh_models.FactorRange(*factors)
        cast(Any, x_range).range_padding = 0.1
        plot = lazy.bokeh_plotting.figure(
            title=f"{metric_column}: {group.name}",
            x_range=x_range,
            y_axis_label=y_axis_label,
            width=800,
            height=500,
            tools="save,reset",
        )
        hover = lazy.bokeh_models.HoverTool(tooltips=[("Node", "@node"), ("Run", "@run"), ("Value", "@value{0.0000}")])
        plot.add_tools(hover)

        if not values:
            return plot

        source = lazy.bokeh_models.ColumnDataSource(
            data={
                "x": factors,
                "node": nodes,
                "run": runs,
                "value": values,
                "color": colors,
            }
        )
        plot.vbar(x="x", top="value", width=0.8, fill_color="color", line_color="color", source=source)
        plot.xaxis.major_label_orientation = 0.8
        y_max = max(values)
        plot.y_range = lazy.bokeh_models.Range1d(start=0, end=y_max * 1.1 if y_max > 0 else 1)
        return plot

    def create_charts(self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            for column in self._available_columns(dfs, self.BANDWIDTH_COLUMNS):
                charts.append(self._create_metric_bar_chart(group, dfs, column, "Bandwidth (GB/s)"))
            for column in self._available_columns(dfs, self.TIME_COLUMNS):
                charts.append(self._create_metric_bar_chart(group, dfs, column, "Time (us)"))
        return charts
