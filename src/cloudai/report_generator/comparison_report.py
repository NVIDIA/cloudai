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

import logging
from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2
from pydantic import Field
from rich.console import Console
from rich.table import Table

from cloudai.core import Reporter, System, TestRun, TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.util.lazy_imports import lazy

from .groups import GroupedTestRuns, TestRunsGrouper
from .util import (
    bokeh_size_unit_js_tick_formatter,
    calculate_power_of_two_ticks,
)

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


class ComparisonReportConfig(ReportConfig):
    """Configuration for a comparison report."""

    enable: bool = True
    group_by: list[str] = Field(default_factory=list)


class ComparisonReport(Reporter, ABC):
    """Base class for comparison reports that generate both charts and tables."""

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.template_path = Path(__file__).parent.parent / "util"
        self.template_name = "nixl_report_template.jinja2"
        self.report_file_name: str = "comparison_report.html"
        self.group_by: list[str] = config.group_by

    @abstractmethod
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame: ...

    @abstractmethod
    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]: ...

    @abstractmethod
    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]: ...

    def get_bokeh_html(self) -> tuple[str, str]:
        cmp_groups = TestRunsGrouper(self.trs, self.group_by).groups()
        charts: list[bk.figure] = self.create_charts(cmp_groups)

        # layout with 2 charts per row
        rows = []
        for i in range(0, len(charts), 2):
            if i + 1 < len(charts):
                rows.append(lazy.bokeh_layouts.row(charts[i], charts[i + 1]))
            else:
                rows.append(lazy.bokeh_layouts.row(charts[i]))
        layout = lazy.bokeh_layouts.column(*rows, name="charts_layout")

        bokeh_script, bokeh_div = lazy.bokeh_embed.components(layout)
        return bokeh_script, bokeh_div

    def generate(self):
        self.load_test_runs()
        if not self.trs:
            logging.debug(f"Skipping {self.__class__.__name__} report generation, no results found.")
            return

        console = Console(record=True)
        cmp_groups = TestRunsGrouper(self.trs, self.group_by).groups()

        tables = self.create_tables(cmp_groups)
        for table in tables:
            console.print(table)
            console.print()

        bokeh_script, bokeh_div = self.get_bokeh_html()

        template = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_path)).get_template(
            self.template_name
        )
        html_content = template.render(
            title=f"{self.test_scenario.name} Comparison Report",
            bokeh_script=bokeh_script,
            bokeh_div=bokeh_div,
            rich_html=console.export_html(),
        )

        html_file = self.results_root / self.report_file_name
        with open(html_file, "w") as f:
            f.write(html_content)

        logging.info(f"Comparison report created: {html_file}")

    @staticmethod
    def _extract_cmp_values(data: list) -> tuple[float | None, float | None]:
        val1, val2 = None, None
        try:
            val1 = float(data[-2])
            val2 = float(data[-1])
        except Exception as e:
            logging.debug(f"Could not extract comparison values from data {data}: {e}")
        return val1, val2

    @staticmethod
    def _format_diff_cell(val1: float | None, val2: float | None) -> str:
        if val1 is None or val2 is None or val2 == 0:
            return "n/a"
        diff = val1 - val2
        diff_percent = (diff / val2) * 100
        return f"{diff:+.2f} ({diff_percent:+.2f}%)"

    def create_table(
        self,
        group: GroupedTestRuns,
        dfs: list[pd.DataFrame],
        title: str,
        info_columns: list[str],
        data_columns: list[str],
    ) -> Table:
        style_cycle = cycle(["green", "cyan", "magenta", "blue", "yellow"])

        table = Table(title=f"{title}: {group.name}", title_justify="left")
        for col in info_columns:
            table.add_column(col)

        enable_diff_column = len(group.items) == 2

        for col in data_columns:
            for item in group.items:
                style = next(style_cycle)
                name_str = "\n".join(item.name.split())
                table.add_column(
                    f"{name_str}\n[white on {style}]{col}",
                    overflow="fold",
                    style=style,
                    header_style=style,
                    no_wrap=False,
                )

            if enable_diff_column:
                diff_style = next(style_cycle)
                table.add_column(f"diff\n{col}", justify="right", style=diff_style, header_style=diff_style)

        df_with_max_rows = max(dfs, key=len)
        for row_idx in range(len(df_with_max_rows)):
            data = []
            for col in data_columns:
                data_points = []
                for df in dfs:
                    data_points.append(str(df[col].get(row_idx, "n/a")))

                if enable_diff_column:
                    val1, val2 = self._extract_cmp_values(data_points)
                    data_points.append(self._format_diff_cell(val1, val2))

                data.extend(data_points)

            table.add_row(*[str(df_with_max_rows[col][row_idx]) for col in info_columns], *data)

        return table

    def create_chart(
        self,
        group: GroupedTestRuns,
        dfs: list[pd.DataFrame],
        title: str,
        info_columns: list[str],
        data_columns: list[str],
        y_axis_label: str,
    ) -> bk.figure:
        style_cycle = cycle(["green", "cyan", "magenta", "blue", "yellow"])

        p = lazy.bokeh_plotting.figure(
            title=f"{title}: {group.name}",
            x_axis_label=info_columns[0],
            y_axis_label=y_axis_label,
            width=800,
            height=500,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            x_axis_type="log",
        )

        hover = lazy.bokeh_models.HoverTool(tooltips=[("X", "@x"), ("Y", "@y"), ("Segment Type", "@segment_type")])
        p.add_tools(hover)

        if all(df.empty for df in dfs):
            logging.debug(f"No data available to create chart for group {group.name}, skipping.")
            return p

        for df, name in zip(dfs, [item.name for item in group.items], strict=True):
            if df.empty:
                continue

            for col in data_columns:
                source = lazy.bokeh_models.ColumnDataSource(
                    data={
                        "x": df[info_columns[0]].tolist(),
                        "y": df[col].tolist(),
                        "segment_type": [col] * len(df),
                    }
                )

                color = next(style_cycle)
                p.line("x", "y", source=source, line_color=color, line_width=2, legend_label=f"{name} {col}")
                p.scatter("x", "y", source=source, fill_color=color, size=8, legend_label=f"{name} {col}")

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        y_max = max(df[col].max() for df in dfs for col in data_columns if not df.empty)
        y_min = min(df[col].min() for df in dfs for col in data_columns if not df.empty)
        p.y_range = lazy.bokeh_models.Range1d(start=y_min * -1 * y_max * 0.01, end=y_max * 1.1)

        df_with_max_rows = max(dfs, key=len)
        x_min = df_with_max_rows[info_columns[0]].min()
        x_max = df_with_max_rows[info_columns[0]].max()
        p.xaxis.ticker = calculate_power_of_two_ticks(x_min, x_max)
        p.xaxis.formatter = lazy.bokeh_models.CustomJSTickFormatter(code=bokeh_size_unit_js_tick_formatter)
        p.xaxis.major_label_orientation = lazy.np.pi / 4

        return p
