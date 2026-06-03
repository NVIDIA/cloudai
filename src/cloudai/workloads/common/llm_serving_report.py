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

import abc
import itertools
import pathlib
from typing import TYPE_CHECKING, Any, cast

import rich.table

import cloudai.core
import cloudai.models.workload
import cloudai.report_generator.comparison_report
import cloudai.report_generator.groups
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd

    from cloudai.workloads.common import llm_serving


class LLMServingComparisonReport(cloudai.report_generator.comparison_report.ComparisonReport, abc.ABC):
    """Shared comparison report for LLM serving benchmarks."""

    LATENCY_METRICS = (
        ("Mean TTFT (ms)", "mean_ttft_ms"),
        ("Median TTFT (ms)", "median_ttft_ms"),
        ("P99 TTFT (ms)", "p99_ttft_ms"),
        ("Mean TPOT (ms)", "mean_tpot_ms"),
        ("Median TPOT (ms)", "median_tpot_ms"),
        ("P99 TPOT (ms)", "p99_tpot_ms"),
    )
    SUCCESS_METRICS = (
        ("Successful Prompts", "completed"),
        ("Successful Prompts (%)", "completion_rate"),
    )
    THROUGHPUT_METRICS = (
        ("Throughput", "throughput"),
        ("TPS/User", "tps_per_user"),
        ("TPS/GPU", "tps_per_gpu"),
    )
    QUALITY_METRICS = (("Accuracy", "accuracy"),)

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self._df_cache: dict[tuple[str, int, int, str], pd.DataFrame] = {}

    @abc.abstractmethod
    def can_handle(self, tr: cloudai.core.TestRun) -> bool:
        """Return whether the report should include the given test run."""

    @abc.abstractmethod
    def parse_results(self, tr: cloudai.core.TestRun) -> tuple[llm_serving.LLMServingBenchReport, int] | None:
        """Parse a workload-specific benchmark result and return it with used GPU count."""

    def parse_accuracy(self, tr: cloudai.core.TestRun) -> float | None:
        """Parse semantic accuracy, if available for this workload run."""
        return None

    @abc.abstractmethod
    def benchmark_cmd_args(self, tr: cloudai.core.TestRun) -> cloudai.models.workload.CmdArgs:
        """Return workload-specific benchmark command arguments for comparison labels."""

    def comparison_values(self, tr: cloudai.core.TestRun) -> dict[str, object]:
        values = super().comparison_values(tr)
        values.update({f"bench_cmd_args.{k}": v for k, v in self.benchmark_cmd_args(tr).model_dump().items()})
        return values

    def load_test_runs(self) -> None:
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if self.can_handle(tr)]

    @staticmethod
    def _metric_value(value: float | int | None) -> float | str:
        if value is None:
            return "n/a"
        return round(float(value), 4)

    @staticmethod
    def _df_cache_key(tr: cloudai.core.TestRun) -> tuple[str, int, int, str]:
        return (tr.name, tr.current_iteration, tr.step, str(tr.output_path))

    def _extract_data_as_df_cached(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
        key = self._df_cache_key(tr)
        if key not in self._df_cache:
            self._df_cache[key] = self.extract_data_as_df(tr)
        return self._df_cache[key]

    def extract_data_as_df(self, tr: cloudai.core.TestRun) -> pd.DataFrame:
        empty_df = lazy.pd.DataFrame(
            {
                "metric_group": lazy.pd.Series([], dtype=str),
                "metric_order": lazy.pd.Series([], dtype=int),
                "metric": lazy.pd.Series([], dtype=str),
                "value": lazy.pd.Series([], dtype=object),
            }
        )
        parsed = self.parse_results(tr)
        if parsed is None:
            return empty_df

        results, used_gpus = parsed
        rows: list[dict[str, Any]] = []
        order = 0

        for label, attr_name in self.LATENCY_METRICS:
            rows.append(
                {
                    "metric_group": "latency",
                    "metric_order": order,
                    "metric": label,
                    "value": self._metric_value(getattr(results, attr_name)),
                }
            )
            order += 1

        completion_rate = None
        if results.num_prompts:
            completion_rate = results.completed / results.num_prompts * 100

        for label, attr_name in self.SUCCESS_METRICS:
            value = completion_rate if attr_name == "completion_rate" else getattr(results, attr_name)
            rows.append(
                {
                    "metric_group": "success",
                    "metric_order": order,
                    "metric": label,
                    "value": self._metric_value(value),
                }
            )
            order += 1

        tps_per_gpu = results.throughput / used_gpus if used_gpus > 0 else None
        for label, attr_name in self.THROUGHPUT_METRICS:
            value = tps_per_gpu if attr_name == "tps_per_gpu" else getattr(results, attr_name)
            rows.append(
                {
                    "metric_group": "throughput",
                    "metric_order": order,
                    "metric": label,
                    "value": self._metric_value(value),
                }
            )
            order += 1

        for label, attr_name in self.QUALITY_METRICS:
            value = self.parse_accuracy(tr) if attr_name == "accuracy" else getattr(results, attr_name)
            rows.append(
                {
                    "metric_group": "quality",
                    "metric_order": order,
                    "metric": label,
                    "value": self._metric_value(value),
                }
            )
            order += 1

        return lazy.pd.DataFrame(rows)

    @staticmethod
    def _group_df(df: pd.DataFrame, metric_group: str) -> pd.DataFrame:
        return (
            df[df["metric_group"] == metric_group]
            .sort_values("metric_order")
            .drop(columns=["metric_group", "metric_order"])
            .reset_index(drop=True)
        )

    def create_tables(
        self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]
    ) -> list[rich.table.Table]:
        tables: list[rich.table.Table] = []
        for group in cmp_groups:
            extracted_dfs = [self._extract_data_as_df_cached(item.tr) for item in group.items]
            tables.extend(
                [
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "latency") for df in extracted_dfs],
                        title="Latency",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "success") for df in extracted_dfs],
                        title="Successful Prompts",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "throughput") for df in extracted_dfs],
                        title="Throughput",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                    self.create_table(
                        group,
                        dfs=[self._group_df(df, "quality") for df in extracted_dfs],
                        title="Quality",
                        info_columns=["metric"],
                        data_columns=["value"],
                    ),
                ]
            )
        return tables

    def _create_metric_bar_chart(
        self,
        group: cloudai.report_generator.groups.GroupedTestRuns,
        dfs: list[pd.DataFrame],
        title: str,
        y_axis_label: str,
    ) -> bk.figure:
        factors: list[tuple[str, str]] = []
        values: list[float] = []
        colors: list[str] = []
        color_cycler = itertools.cycle(["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"])
        color_by_run = {item.name: next(color_cycler) for item in group.items}

        for df, run_name in zip(dfs, [item.name for item in group.items], strict=True):
            for _, row in df.iterrows():
                value = row["value"]
                if not isinstance(value, (float, int)):
                    continue
                factors.append((row["metric"], run_name))
                values.append(float(value))
                colors.append(color_by_run[run_name])

        x_range = lazy.bokeh_models.FactorRange(*factors)
        cast(Any, x_range).range_padding = 0.1
        p = lazy.bokeh_plotting.figure(
            title=f"{title}: {group.name}",
            x_range=x_range,
            y_axis_label=y_axis_label,
            width=800,
            height=500,
            tools="save,reset",
        )
        hover = lazy.bokeh_models.HoverTool(
            tooltips=[("Metric", "@metric"), ("Run", "@run"), ("Value", "@value{0.0000}")]
        )
        p.add_tools(hover)

        if not values:
            return p

        source = lazy.bokeh_models.ColumnDataSource(
            data={
                "x": factors,
                "metric": [metric for metric, _ in factors],
                "run": [run for _, run in factors],
                "value": values,
                "color": colors,
            }
        )
        p.vbar(x="x", top="value", width=0.8, fill_color="color", line_color="color", source=source)
        p.xaxis.major_label_orientation = 0.8
        p.y_range = lazy.bokeh_models.Range1d(start=0, end=max(values) * 1.1)
        return p

    def create_charts(self, cmp_groups: list[cloudai.report_generator.groups.GroupedTestRuns]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for group in cmp_groups:
            extracted_dfs = [self._extract_data_as_df_cached(item.tr) for item in group.items]
            charts.extend(
                [
                    self._create_metric_bar_chart(
                        group,
                        [self._group_df(df, "latency") for df in extracted_dfs],
                        "Latency",
                        "Latency (ms)",
                    ),
                    self._create_metric_bar_chart(
                        group,
                        [self._group_df(df, "success") for df in extracted_dfs],
                        "Successful Prompts",
                        "Prompts / %",
                    ),
                    self._create_metric_bar_chart(
                        group,
                        [self._group_df(df, "throughput") for df in extracted_dfs],
                        "Throughput",
                        "Throughput",
                    ),
                    self._create_metric_bar_chart(
                        group,
                        [self._group_df(df, "quality") for df in extracted_dfs],
                        "Quality",
                        "Score",
                    ),
                ]
            )
        return charts
