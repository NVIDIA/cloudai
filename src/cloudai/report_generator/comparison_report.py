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

import collections
import collections.abc
import dataclasses
import logging
from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import jinja2
import yaml
from pydantic import Field
from rich.console import Console
from rich.table import Table

import cloudai.metrics
from cloudai.core import Reporter, System, TestRun, TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.util.lazy_imports import lazy

from .groups import GroupedTestRuns, TRGroupItem
from .util import (
    bokeh_size_unit_js_tick_formatter,
    calculate_power_of_two_ticks,
    diff_comparison_values,
)

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


@dataclasses.dataclass
class MetricColumn:
    """Declare which canonical metric and coordinates a dataframe column represents."""

    metric: cloudai.metrics.MetricDefinition
    coordinate_columns: dict[str, str] = dataclasses.field(default_factory=dict)
    coordinate_values: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ComparisonSection:
    """Normalized comparison data consumed by both report renderers."""

    group: GroupedTestRuns
    title: str
    dfs: list[pd.DataFrame]
    info_columns: list[str]
    data_columns: list[str]
    y_axis_label: str
    chart_type: Literal["line", "bar"] = "line"
    x_axis_type: Literal["linear", "logarithmic", "category", "indexed_category"] = "logarithmic"
    x_axis_column: str | None = None
    x_axis_label: str | None = None
    y_axis_type: Literal["linear", "logarithmic", "auto"] = "linear"
    metric_columns: dict[str, MetricColumn] = dataclasses.field(default_factory=dict)


class _IndentedSafeDumper(yaml.SafeDumper):
    """Indent sequence items beneath their mapping key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        del indentless  # Required by PyYAML's keyword-call interface; indentation is always enabled.
        super().increase_indent(flow, indentless=False)


class ComparisonReportConfig(ReportConfig):
    """Configuration for a comparison report."""

    enable: bool = True
    group_by: list[str] = Field(default_factory=list)


class ComparisonReport(Reporter, ABC):
    """Base class for comparison reports that generate both charts and tables."""

    SOL_REFERENCE_COLOR = "#741D9D"

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.template_path = Path(__file__).parent.parent / "util"
        self.template_name = "nixl_report_template.jinja2"
        self.template_name_v2 = "comparison-report-v2.jinja2"
        self.report_file_name: str = "comparison_report.html"
        self.group_by: list[str] = config.group_by
        self._metric_assessment_cache: dict[int, list[cloudai.metrics.MetricAssessment]] = {}

    def _assessments(self, tr: TestRun) -> list[cloudai.metrics.MetricAssessment]:
        cache_key = id(tr)
        if cache_key not in self._metric_assessment_cache:
            self._metric_assessment_cache[cache_key] = cloudai.metrics.assess_test_run_metrics(self.system, tr)
        return self._metric_assessment_cache[cache_key]

    @staticmethod
    def _coordinate_matches(actual: object, expected: object) -> bool:
        if actual == expected:
            return True
        try:
            return float(cast(Any, actual)) == float(cast(Any, expected))
        except (TypeError, ValueError):
            return str(actual) == str(expected)

    def _assessment_for_row(
        self, section: ComparisonSection, data_column: str, item_idx: int, row_idx: int
    ) -> cloudai.metrics.MetricAssessment | None:
        binding = section.metric_columns.get(data_column)
        if binding is None or row_idx >= len(section.dfs[item_idx]):
            return None
        df = section.dfs[item_idx]
        row = df.iloc[row_idx]
        expected = {
            **binding.coordinate_values,
            **{name: row[column] for name, column in binding.coordinate_columns.items()},
        }
        for assessment in self._assessments(section.group.items[item_idx].tr):
            if assessment.observation.metric is not binding.metric:
                continue
            coordinates = cloudai.metrics.coordinates_dict(assessment.observation.coordinates)
            if all(self._coordinate_matches(coordinates.get(name), value) for name, value in expected.items()):
                return assessment
        return None

    def _column_has_sol(self, section: ComparisonSection, data_column: str) -> bool:
        return any(
            (assessment := self._assessment_for_row(section, data_column, item_idx, row_idx)) is not None
            and assessment.sol is not None
            for item_idx, df in enumerate(section.dfs)
            for row_idx in range(len(df))
        )

    def _shared_sol_curve(
        self, section: ComparisonSection, data_column: str, x_column: str
    ) -> list[tuple[Any, float]] | None:
        """Return the SOL curve when every compared item resolves the same points."""
        curves: list[list[tuple[Any, float]]] = []
        for item_idx, df in enumerate(section.dfs):
            curve: list[tuple[Any, float]] = []
            for row_idx in range(len(df)):
                assessment = self._assessment_for_row(section, data_column, item_idx, row_idx)
                if assessment is not None and assessment.sol is not None:
                    curve.append((df[x_column].iloc[row_idx], assessment.sol))
            if not curve:
                return None
            curves.append(curve)

        reference = curves[0]
        if any(
            len(curve) != len(reference)
            or any(
                not self._coordinate_matches(actual_x, expected_x)
                or not self._coordinate_matches(actual_sol, expected_sol)
                for (actual_x, actual_sol), (expected_x, expected_sol) in zip(curve, reference, strict=True)
            )
            for curve in curves[1:]
        ):
            return None
        return reference

    @abstractmethod
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame: ...

    @abstractmethod
    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
        """Return normalized sections without performing any rendering."""

    def comparison_values(self, tr: TestRun) -> dict[str, object]:
        """Return TestRun values used to label differences between compared runs."""
        return {
            "NUM_NODES": tr.num_nodes,
            **tr.test.cmd_args.model_dump(exclude_none=True),
            **{f"extra_env_vars.{key}": value for key, value in tr.test.extra_env_vars.items()},
        }

    def get_group_value(self, tr: TestRun, field: str) -> str:
        """Get grouping field value for a TestRun's cmd_args or extra_env_vars."""
        if field.startswith("extra_env_vars."):
            field_name = field[len("extra_env_vars.") :]
            return str(tr.test.extra_env_vars.get(field_name))
        return getattr(tr.test.cmd_args, field)

    def group_name(self, trs: list[TestRun]) -> str:
        """Return display name for a group of TestRuns."""
        if not self.group_by:
            return "all-in-one"
        parts = [f"{field}={self.get_group_value(trs[0], field)}" for field in self.group_by]
        return " ".join(parts).replace("extra_env_vars.", "")

    def create_group(self, trs: list[TestRun], group_idx: str = "0") -> GroupedTestRuns:
        """Create a comparison group using report-specific comparison values."""
        diff = diff_comparison_values([self.comparison_values(tr) for tr in trs])
        compact_names = [self._compact_case_name(tr) for tr in trs]
        duplicate_names = collections.Counter(compact_names)
        duplicate_indexes: collections.defaultdict[str, int] = collections.defaultdict(int)
        items: list[TRGroupItem] = []
        for idx, tr in enumerate(trs):
            name = f"{group_idx}.{idx}"
            if diff:
                item_name_parts = [f"{field}={vals[idx]}" for field, vals in diff.items()]
                name = " ".join(item_name_parts).replace("extra_env_vars.", "")

            compact_name = compact_names[idx]
            if duplicate_names[compact_name] > 1:
                duplicate_indexes[compact_name] += 1
                compact_name = f"{compact_name} · run={duplicate_indexes[compact_name]}"

            items.append(
                TRGroupItem(
                    name=name,
                    tr=tr,
                    compact_name=compact_name,
                    differences=self._structured_diff(diff, idx),
                )
            )
        return GroupedTestRuns(name=self.group_name(trs), items=items)

    @staticmethod
    def _compact_case_name(tr: TestRun) -> str:
        parts = [tr.name]
        if tr.step > 0:
            parts.append(f"step={tr.step}")
        if tr.iterations > 1:
            parts.append(f"iter={tr.current_iteration}")
        return " · ".join(parts)

    @classmethod
    def _structured_diff(cls, diff: dict[str, list[object]], item_idx: int) -> dict[str, Any]:
        """Return one run's differing parameters as a nested mapping."""
        result: dict[str, Any] = {}
        for field, values in diff.items():
            value = values[item_idx]
            if value is None:
                continue
            path = field.split(".")
            target = result
            for key in path[:-1]:
                nested = target.get(key)
                if not isinstance(nested, dict):
                    nested = {}
                    target[key] = nested
                target = nested
            target[path[-1]] = value
        return result

    @staticmethod
    def _format_diff_yaml(differences: collections.abc.Mapping[str, object] | None) -> str:
        if not differences:
            return ""
        return yaml.dump(
            differences,
            Dumper=_IndentedSafeDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).rstrip()

    def group_test_runs(self) -> list[GroupedTestRuns]:
        """Group loaded TestRuns for this comparison report."""
        if not self.group_by:
            return [self.create_group(self.trs)]

        groups: list[list[TestRun]] = []
        for tr in self.trs:
            for group in groups:
                matched = all(
                    self.get_group_value(tr, field) == self.get_group_value(group[0], field) for field in self.group_by
                )

                if matched:
                    group.append(tr)
                    break
            else:
                groups.append([tr])

        return [self.create_group(group, group_idx=str(group_idx)) for group_idx, group in enumerate(groups)]

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        """Render legacy Rich tables from normalized sections."""
        return [
            self.create_table(
                section.group,
                section.dfs,
                section.title,
                section.info_columns,
                section.data_columns,
                section=section,
            )
            for section in self.build_sections(cmp_groups)
        ]

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        """Render legacy Bokeh charts from normalized sections."""
        return self._render_bokeh_charts(self.build_sections(cmp_groups))

    def _render_bokeh_charts(self, sections: list[ComparisonSection]) -> list[bk.figure]:
        charts: list[bk.figure] = []
        for section in sections:
            if section.chart_type == "bar":
                charts.append(self.create_bar_chart(section))
            else:
                charts.append(
                    self.create_chart(
                        section.group,
                        section.dfs,
                        section.title,
                        section.info_columns,
                        section.data_columns,
                        section.y_axis_label,
                        section=section,
                    )
                )
        return charts

    def get_bokeh_html(self, sections: list[ComparisonSection] | None = None) -> tuple[str, str]:
        if sections is None:
            sections = self.build_sections(self.group_test_runs())
        charts = self._render_bokeh_charts(sections)

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

    def _render_console(self, sections: list[ComparisonSection]) -> str:
        console = Console(record=True)
        for section in sections:
            table = self.create_table(
                section.group,
                section.dfs,
                section.title,
                section.info_columns,
                section.data_columns,
                section=section,
            )
            console.print(table)
            console.print()
        return console.export_html()

    def _render_html(
        self,
        env: jinja2.Environment,
        sections: list[ComparisonSection],
        console_html: str,
    ) -> str:
        bokeh_script, bokeh_div = self.get_bokeh_html(sections)
        template = env.get_template(self.template_name)
        return template.render(
            title=f"{self.test_scenario.name} Comparison Report",
            bokeh_script=bokeh_script,
            bokeh_div=bokeh_div,
            rich_html=console_html,
        )

    def _render_html_v2(self, env: jinja2.Environment, sections: list[ComparisonSection]) -> str:
        template_v2 = env.get_template(self.template_name_v2)
        return template_v2.render(
            name=f"{self.test_scenario.name} Comparison Report",
            sections=self._build_sections_v2(sections),
        )

    def generate(self):
        self.load_test_runs()
        if not self.trs:
            logging.debug(f"Skipping {self.__class__.__name__} report generation, no results found.")
            return

        sections = self.build_sections(self.group_test_runs())
        console_html = self._render_console(sections)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_path),
            autoescape=jinja2.select_autoescape(enabled_extensions=("html", "htm", "xml", "jinja2")),
        )
        rendered_reports = [
            ("Comparison report", self.report_file_name, self._render_html(env, sections, console_html)),
            ("Comparison report v2", self.report_file_name_v2, self._render_html_v2(env, sections)),
        ]

        for report_name, file_name, content in rendered_reports:
            report_path = self.results_root / file_name
            report_path.write_text(content, encoding="utf-8")
            logging.info(f"{report_name} created: {report_path}")

    @property
    def report_file_name_v2(self) -> str:
        report_path = Path(self.report_file_name)
        return f"{report_path.stem}_v2{report_path.suffix}"

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

    @staticmethod
    def _display_value(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            if lazy.pd.isna(value):
                return "n/a"
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _numeric_value(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not lazy.np.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _chart_label(item: TRGroupItem, data_column: str, include_metric: bool) -> str:
        return f"{item.compact_name_v2} · {data_column}" if include_metric else item.compact_name_v2

    def _build_bar_datasets_v2(self, section: ComparisonSection) -> tuple[list[str], list[dict[str, Any]]]:
        widest_df = max(section.dfs, key=len)
        labels = [self._display_value(value) for value in widest_df[section.info_columns[0]].tolist()]
        datasets: list[dict[str, Any]] = []
        include_metric = len(section.data_columns) > 1
        for data_column in section.data_columns:
            for item, df in zip(section.group.items, section.dfs, strict=True):
                datasets.append(
                    {
                        "label": self._chart_label(item, data_column, include_metric),
                        "data": [
                            self._numeric_value(df[data_column].get(row_idx, None)) for row_idx in range(len(labels))
                        ],
                    }
                )
        return labels, datasets

    def _build_line_datasets_v2(self, section: ComparisonSection) -> tuple[list[str] | None, list[dict[str, Any]]]:
        datasets: list[dict[str, Any]] = []
        include_metric = len(section.data_columns) > 1
        x_column = section.x_axis_column or section.info_columns[0]
        labels: list[str] | None = None
        if section.x_axis_type in ("category", "indexed_category"):
            widest_df = max(section.dfs, key=len)
            labels = [self._display_value(value) for value in widest_df[x_column].tolist()]

        series_idx = 0
        for data_column in section.data_columns:
            for item, df in zip(section.group.items, section.dfs, strict=True):
                if labels is not None:
                    data: list[float | None] | list[dict[str, float]] = [
                        self._numeric_value(df[data_column].get(row_idx, None)) for row_idx in range(len(labels))
                    ]
                else:
                    points: list[dict[str, float]] = []
                    pairs = (
                        zip(
                            df[x_column].tolist(),
                            df[data_column].tolist(),
                            strict=True,
                        )
                        if not df.empty and data_column in df
                        else []
                    )
                    for x_value, y_value in pairs:
                        numeric_x = self._numeric_value(x_value)
                        numeric_y = self._numeric_value(y_value)
                        if numeric_x is not None and numeric_y is not None:
                            points.append({"x": numeric_x, "y": numeric_y})
                    data = points
                datasets.append(
                    {
                        "label": self._chart_label(item, data_column, include_metric),
                        "data": data,
                        "source_color_index": series_idx,
                    }
                )
                series_idx += 1

            if (sol_curve := self._shared_sol_curve(section, data_column, x_column)) is None:
                continue
            if labels is None:
                sol_data = [
                    {"x": numeric_x, "y": sol}
                    for x_value, sol in sol_curve
                    if (numeric_x := self._numeric_value(x_value)) is not None
                ]
            else:
                sol_by_label = {self._display_value(x_value): sol for x_value, sol in sol_curve}
                sol_data = [sol_by_label.get(label) for label in labels]
            datasets.append(
                {
                    "label": f"{data_column} · SOL" if include_metric else "SOL",
                    "data": sol_data,
                    "is_sol": True,
                }
            )
        return labels, datasets

    def _build_chart_v2(self, section: ComparisonSection, chart_idx: int) -> dict[str, Any]:
        if section.chart_type == "bar":
            chart_labels, datasets = self._build_bar_datasets_v2(section)
            x_axis_type = "category"
        else:
            chart_labels, datasets = self._build_line_datasets_v2(section)
            x_axis_type = section.x_axis_type

        return {
            "id": f"comparison-chart-{chart_idx}",
            "type": section.chart_type,
            "labels": chart_labels,
            "datasets": datasets,
            "sol_color": self.SOL_REFERENCE_COLOR,
            "x_axis_label": section.x_axis_label or section.x_axis_column or section.info_columns[0],
            "x_axis_type": x_axis_type,
            "y_axis_label": section.y_axis_label,
            "y_axis_type": section.y_axis_type,
        }

    def _build_table_v2(self, section: ComparisonSection) -> dict[str, Any]:
        widest_df = max(section.dfs, key=len)
        show_diff = len(section.group.items) == 2
        data_headers: list[dict[str, str]] = []
        for data_column in section.data_columns:
            show_sol = self._column_has_sol(section, data_column)
            for item in section.group.items:
                data_headers.append(
                    {
                        "name": item.compact_name_v2,
                        "differences_yaml": self._format_diff_yaml(item.differences),
                        "metric": data_column,
                    }
                )
                if show_sol:
                    data_headers.extend(
                        [
                            {"name": f"{item.compact_name_v2} · SOL", "differences_yaml": "", "metric": data_column},
                            {"name": f"{item.compact_name_v2} · % SOL", "differences_yaml": "", "metric": data_column},
                        ]
                    )
            if show_diff:
                data_headers.append(
                    {
                        "name": "Difference",
                        "differences_yaml": "",
                        "metric": data_column,
                    }
                )

        rows: list[dict[str, list[str]]] = []
        for row_idx in range(len(widest_df)):
            info_cells = [self._display_value(widest_df[column].get(row_idx, None)) for column in section.info_columns]
            data_cells: list[str] = []
            for data_column in section.data_columns:
                raw_values = [df[data_column].get(row_idx, None) for df in section.dfs]
                show_sol = self._column_has_sol(section, data_column)
                for item_idx, value in enumerate(raw_values):
                    data_cells.append(self._display_value(value))
                    if show_sol:
                        assessment = self._assessment_for_row(section, data_column, item_idx, row_idx)
                        data_cells.extend(
                            [
                                self._display_value(assessment.sol if assessment else None),
                                f"{assessment.attainment:.1%}"
                                if assessment is not None and assessment.attainment is not None
                                else "n/a",
                            ]
                        )
                if show_diff:
                    val1, val2 = self._extract_cmp_values(raw_values)
                    data_cells.append(self._format_diff_cell(val1, val2))
            rows.append({"info_cells": info_cells, "data_cells": data_cells})

        return {
            "info_headers": section.info_columns,
            "data_headers": data_headers,
            "rows": rows,
        }

    def _build_sections_v2(self, sections: list[ComparisonSection]) -> list[dict[str, Any]]:
        return [
            {
                "title": section.title,
                "group_name": "All cases" if section.group.name == "all-in-one" else section.group.name,
                "chart": self._build_chart_v2(section, idx),
                "table": self._build_table_v2(section),
                "case_details": [
                    {
                        "name": item.compact_name_v2,
                        "differences_yaml": self._format_diff_yaml(item.differences),
                    }
                    for item in section.group.items
                ],
            }
            for idx, section in enumerate(sections)
        ]

    def _rich_data_points(
        self,
        section: ComparisonSection | None,
        data_column: str,
        dfs: list[pd.DataFrame],
        row_idx: int,
        enable_diff_column: bool,
    ) -> list[str]:
        raw_values = [df[data_column].get(row_idx, "n/a") for df in dfs]
        show_sol = section is not None and self._column_has_sol(section, data_column)
        data_points: list[str] = []
        for item_idx, value in enumerate(raw_values):
            data_points.append(str(value))
            if show_sol and section is not None:
                assessment = self._assessment_for_row(section, data_column, item_idx, row_idx)
                data_points.extend(
                    [
                        self._display_value(assessment.sol if assessment else None),
                        f"{assessment.attainment:.1%}"
                        if assessment is not None and assessment.attainment is not None
                        else "n/a",
                    ]
                )
        if enable_diff_column:
            val1, val2 = self._extract_cmp_values(raw_values)
            data_points.append(self._format_diff_cell(val1, val2))
        return data_points

    def create_table(
        self,
        group: GroupedTestRuns,
        dfs: list[pd.DataFrame],
        title: str,
        info_columns: list[str],
        data_columns: list[str],
        section: ComparisonSection | None = None,
    ) -> Table:
        style_cycle = cycle(["green", "cyan", "magenta", "blue", "yellow"])

        table = Table(title=f"{title}: {group.name}", title_justify="left")
        for col in info_columns:
            table.add_column(col)

        enable_diff_column = len(group.items) == 2

        for col in data_columns:
            show_sol = section is not None and self._column_has_sol(section, col)
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
                if show_sol:
                    table.add_column(f"{name_str}\nSOL", overflow="fold", style=style, header_style=style)
                    table.add_column(f"{name_str}\n% SOL", overflow="fold", style=style, header_style=style)

            if enable_diff_column:
                diff_style = next(style_cycle)
                table.add_column(f"diff\n{col}", justify="right", style=diff_style, header_style=diff_style)

        df_with_max_rows = max(dfs, key=len)
        for row_idx in range(len(df_with_max_rows)):
            data = []
            for col in data_columns:
                data.extend(self._rich_data_points(section, col, dfs, row_idx, enable_diff_column))

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
        section: ComparisonSection | None = None,
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

        sol_y_values: list[float] = []
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

        if section is not None:
            for col in data_columns:
                sol_points = self._shared_sol_curve(section, col, info_columns[0])
                if sol_points is None:
                    continue
                sol_y_values.extend(point[1] for point in sol_points)
                sol_label = f"{col} SOL" if len(data_columns) > 1 else "SOL"
                p.line(
                    [point[0] for point in sol_points],
                    [point[1] for point in sol_points],
                    line_color=self.SOL_REFERENCE_COLOR,
                    line_dash="dashed",
                    line_width=3,
                    legend_label=sol_label,
                )
                p.scatter(
                    [point[0] for point in sol_points],
                    [point[1] for point in sol_points],
                    marker="diamond",
                    fill_color=self.SOL_REFERENCE_COLOR,
                    line_color="#ffffff",
                    line_width=1.5,
                    size=10,
                    legend_label=sol_label,
                )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        measured_y_values = [df[col].max() for df in dfs for col in data_columns if not df.empty]
        measured_y_min_values = [df[col].min() for df in dfs for col in data_columns if not df.empty]
        y_max = max([*measured_y_values, *sol_y_values])
        y_min = min([*measured_y_min_values, *sol_y_values])
        p.y_range = lazy.bokeh_models.Range1d(start=y_min * -1 * y_max * 0.01, end=y_max * 1.1)

        df_with_max_rows = max(dfs, key=len)
        x_min = df_with_max_rows[info_columns[0]].min()
        x_max = df_with_max_rows[info_columns[0]].max()
        p.xaxis.ticker = calculate_power_of_two_ticks(x_min, x_max)
        p.xaxis.formatter = lazy.bokeh_models.CustomJSTickFormatter(code=bokeh_size_unit_js_tick_formatter)
        p.xaxis.major_label_orientation = lazy.np.pi / 4

        return p

    def create_bar_chart(self, section: ComparisonSection) -> bk.figure:
        """Render the legacy grouped bar chart used by categorical sections."""
        factors: list[tuple[str, str]] = []
        values: list[float] = []
        categories: list[str] = []
        runs: list[str] = []
        colors: list[str] = []
        color_cycle = cycle(["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"])
        color_by_run = {item.name: next(color_cycle) for item in section.group.items}
        info_column = section.info_columns[0]

        for df, item in zip(section.dfs, section.group.items, strict=True):
            for _, row in df.iterrows():
                category = str(row[info_column])
                for data_column in section.data_columns:
                    value = self._numeric_value(row[data_column])
                    if value is None:
                        continue
                    factor_category = category if len(section.data_columns) == 1 else f"{category} {data_column}"
                    factors.append((factor_category, item.name))
                    values.append(value)
                    categories.append(factor_category)
                    runs.append(item.name)
                    colors.append(color_by_run[item.name])

        x_range = lazy.bokeh_models.FactorRange(*factors)
        x_range.range_padding = 0.1
        plot = lazy.bokeh_plotting.figure(
            title=f"{section.title}: {section.group.name}",
            x_range=x_range,
            y_axis_label=section.y_axis_label,
            width=800,
            height=500,
            tools="save,reset",
        )
        hover = lazy.bokeh_models.HoverTool(
            tooltips=[
                (info_column.title(), "@category"),
                ("Run", "@run"),
                ("Value", "@value{0.0000}"),
            ]
        )
        plot.add_tools(hover)

        if not values:
            return plot

        source = lazy.bokeh_models.ColumnDataSource(
            data={
                "x": factors,
                "category": categories,
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
