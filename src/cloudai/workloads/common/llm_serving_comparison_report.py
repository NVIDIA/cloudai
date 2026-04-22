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

from abc import ABC, abstractmethod
from itertools import cycle
from typing import TYPE_CHECKING, Any

from rich.table import Table

from cloudai.core import TestRun
from cloudai.report_generator.comparison_report import ComparisonReport
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class LLMServingComparisonReport(ComparisonReport, ABC):
    """Shared comparison report for LLM serving benchmarks."""

    LATENCY_METRICS = (
        ("Mean TTFT (ms)", "mean_ttft_ms"),
        ("Median TTFT (ms)", "median_ttft_ms"),
        ("P99 TTFT (ms)", "p99_ttft_ms"),
        ("Mean TPOT (ms)", "mean_tpot_ms"),
        ("Median TPOT (ms)", "median_tpot_ms"),
        ("P99 TPOT (ms)", "p99_tpot_ms"),
    )
    THROUGHPUT_METRICS = (
        ("Throughput", "throughput"),
        ("TPS/User", "tps_per_user"),
        ("TPS/GPU", "tps_per_gpu"),
    )

    @abstractmethod
    def can_handle(self, tr: TestRun) -> bool:
        """Return whether the report should include the given test run."""

    @abstractmethod
    def parse_results(self, tr: TestRun) -> tuple[Any, int] | None:
        """Parse a workload-specific benchmark result and return it with used GPU count."""

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if self.can_handle(tr)]

    @staticmethod
    def _metric_value(value: float | int | None) -> float | str:
        if value is None:
            return "n/a"
        return round(float(value), 4)

    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        parsed = self.parse_results(tr)
        if parsed is None:
            return lazy.pd.DataFrame(
                {
                    "metric_group": lazy.pd.Series([], dtype=str),
                    "metric_order": lazy.pd.Series([], dtype=int),
                    "metric": lazy.pd.Series([], dtype=str),
                    "value": lazy.pd.Series([], dtype=object),
                }
            )

        results, used_gpus = parsed
        rows: list[dict[str, object]] = []
        for idx, (label, attr_name) in enumerate(self.LATENCY_METRICS):
            rows.append(
                {
                    "metric_group": "latency",
                    "metric_order": idx,
                    "metric": label,
                    "value": self._metric_value(getattr(results, attr_name)),
                }
            )

        for idx, (label, attr_name) in enumerate(self.THROUGHPUT_METRICS):
            if attr_name == "tps_per_gpu":
                value = results.throughput / used_gpus if used_gpus > 0 else None
            else:
                value = getattr(results, attr_name)
            rows.append(
                {
                    "metric_group": "throughput",
                    "metric_order": idx,
                    "metric": label,
                    "value": self._metric_value(value),
                }
            )

        return lazy.pd.DataFrame(rows)

    @staticmethod
    def _group_df(df: pd.DataFrame, metric_group: str) -> pd.DataFrame:
        return df[df["metric_group"] == metric_group].reset_index(drop=True)

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        tables: list[Table] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            tables.extend(
                [
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "latency") for df in dfs],
                        title="Latency",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "throughput") for df in dfs],
                        title="Throughput",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                ]
            )
        return tables

    def _create_metric_bar_chart(
        self,
        group: GroupedTestRuns,
        dfs: list[pd.DataFrame],
        metric_group: str,
        title: str,
        y_axis_label: str,
    ) -> bk.figure:
        factors: list[tuple[str, str]] = []
        values: list[float] = []
        colors: list[str] = []
        metric_names: list[str] = []
        run_names: list[str] = []
        color_cycle = cycle(["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"])

        for item, df in zip(group.items, dfs, strict=True):
            metric_df = self._group_df(df, metric_group)
            color = next(color_cycle)
            for _, row in metric_df.iterrows():
                value = row["value"]
                if isinstance(value, str):
                    continue
                metric = str(row["metric"])
                factors.append((metric, item.name))
                values.append(float(value))
                colors.append(color)
                metric_names.append(metric)
                run_names.append(item.name)

        p = lazy.bokeh_plotting.figure(
            x_range=lazy.bokeh_models.FactorRange(*factors),
            title=f"{title}: {group.name}",
            y_axis_label=y_axis_label,
            width=900,
            height=500,
            tools="save,reset",
        )
        p.add_tools(
            lazy.bokeh_models.HoverTool(tooltips=[("Metric", "@metric"), ("Run", "@run"), ("Value", "@value{0.0000}")])
        )

        if not factors:
            return p

        source = lazy.bokeh_models.ColumnDataSource(
            data={
                "x": factors,
                "value": values,
                "color": colors,
                "metric": metric_names,
                "run": run_names,
            }
        )
        p.vbar(x="x", top="value", width=0.8, fill_color="color", line_color="color", source=source)
        p.x_range.range_padding = 0.05
        p.xaxis.major_label_orientation = 1.0
        p.y_range.start = 0
        p.y_range.end = max(values) * 1.1 if values else 1

        return p

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            dfs = [self.extract_data_as_df(item.tr) for item in group.items]
            charts.extend(
                [
                    self._create_metric_bar_chart(group, dfs, "latency", "Latency", "Latency (ms)"),
                    self._create_metric_bar_chart(group, dfs, "throughput", "Throughput", "Throughput"),
                ]
            )
        return charts
