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

"""Opt-in SOL support for metric-based comparison reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cloudai.metrics
from cloudai.core import TestRun
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonSection
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class SOLComparisonReport(ComparisonReport):
    """Build explicitly configured SOL metric comparisons."""

    SOL_REFERENCE_COLOR = "#741D9D"

    def _assessments(self, tr: TestRun) -> list[cloudai.metrics.MetricAssessment]:
        return cloudai.metrics.assess_test_run_metrics(self.system, tr)

    @classmethod
    def _sol_column(cls, data_column: str) -> str:
        return f"{data_column} SOL"

    @classmethod
    def _attainment_column(cls, data_column: str) -> str:
        return f"{data_column} % SOL"

    def build_metric_section(
        self,
        group: GroupedTestRuns,
        metric: cloudai.metrics.MetricDefinition,
        x_dimension: str,
        series_dimensions: tuple[str, ...] = (),
    ) -> ComparisonSection | None:
        """Build one explicitly configured metric section for a comparison group."""
        frames = [self._assessment_frame(self._assessments(item.tr), metric) for item in group.items]
        if all(frame.empty for frame in frames):
            return None
        return self._build_curve_section(group, metric, x_dimension, series_dimensions, frames)

    @staticmethod
    def _assessment_frame(
        assessments: list[cloudai.metrics.MetricAssessment],
        metric: cloudai.metrics.MetricDefinition,
    ) -> pd.DataFrame:
        """Convert one run's assessments for a metric into a dataframe."""
        return lazy.pd.DataFrame(
            [
                {
                    **assessment.observation.dimensions,
                    "measured": assessment.observation.value,
                    "sol": assessment.sol,
                    "attainment": assessment.attainment,
                }
                for assessment in assessments
                if assessment.observation.metric is metric
            ]
        )

    def _build_curve_section(
        self,
        group: GroupedTestRuns,
        metric: cloudai.metrics.MetricDefinition,
        x_dimension: str,
        series_dimensions: tuple[str, ...],
        frames: list[pd.DataFrame],
    ) -> ComparisonSection:
        """Build measured and SOL curves using explicit axis and series dimensions."""
        series = list(series_dimensions)
        series_keys = list(
            dict.fromkeys(
                tuple(row)
                for frame in frames
                if not frame.empty
                for row in (frame[series].drop_duplicates().itertuples(index=False, name=None) if series else [()])
            )
        )
        data_columns = [
            " · ".join(
                f"{cloudai.metrics.dimension_label(dimension)}={cloudai.metrics.format_dimension(dimension, value)}"
                for dimension, value in zip(series, key, strict=True)
            )
            or metric.display_name
            for key in series_keys
        ]

        x_column = cloudai.metrics.dimension_label(x_dimension)
        x_label_column = f"{x_column} label"
        dfs = []
        for frame in frames:
            x_values = sorted(frame[x_dimension].unique()) if not frame.empty else []
            result = lazy.pd.DataFrame(
                {
                    x_column: x_values,
                    x_label_column: [cloudai.metrics.format_dimension(x_dimension, value) for value in x_values],
                }
            )
            for series_key, data_column in zip(series_keys, data_columns, strict=True):
                if frame.empty:
                    result[data_column] = None
                    result[self._sol_column(data_column)] = None
                    result[self._attainment_column(data_column)] = None
                    continue
                points = frame
                for dimension, value in zip(series, series_key, strict=True):
                    points = points[points[dimension] == value]
                points = points.groupby(x_dimension, as_index=False).first().set_index(x_dimension)
                result[data_column] = result[x_column].map(points["measured"])
                result[self._sol_column(data_column)] = result[x_column].map(points["sol"])
                result[self._attainment_column(data_column)] = result[x_column].map(points["attainment"])
            dfs.append(result)

        return ComparisonSection(
            group=group,
            title=metric.display_name,
            dfs=dfs,
            info_columns=[x_column],
            data_columns=data_columns,
            y_axis_label=f"{metric.display_name} ({metric.unit})",
            x_axis_type="indexed_category",
            x_axis_column=x_label_column,
            x_axis_label=x_column,
        )

    def _column_has_sol(self, section: ComparisonSection, data_column: str) -> bool:
        sol_column = self._sol_column(data_column)
        return any(sol_column in df and df[sol_column].notna().any() for df in section.dfs)

    def _shared_sol_curve(
        self, section: ComparisonSection, data_column: str, x_column: str
    ) -> list[tuple[Any, float]] | None:
        """Return a SOL curve only when it is identical for every compared run."""
        sol_column = self._sol_column(data_column)
        curves = []
        for df in section.dfs:
            if x_column not in df or sol_column not in df:
                return None
            curve = [
                (x_value, float(sol))
                for x_value, sol in zip(df[x_column], df[sol_column], strict=True)
                if not lazy.pd.isna(sol)
            ]
            if not curve:
                return None
            curves.append(curve)

        reference = curves[0]
        return reference if all(curve == reference for curve in curves[1:]) else None

    def _build_line_datasets_v2(self, section: ComparisonSection) -> tuple[list[str] | None, list[dict[str, Any]]]:
        labels, datasets = super()._build_line_datasets_v2(section)
        if labels is None:
            raise ValueError("SOL comparison charts require a categorical x-axis")
        include_metric = len(section.data_columns) > 1
        x_column = section.x_axis_column or section.info_columns[0]
        inserted = 0
        for metric_idx, data_column in enumerate(section.data_columns):
            sol_curve = self._shared_sol_curve(section, data_column, x_column)
            if sol_curve is None:
                continue
            sol_by_label = {self._display_value(x_value): sol for x_value, sol in sol_curve}
            sol_data = [sol_by_label.get(label) for label in labels]
            insert_at = (metric_idx + 1) * len(section.group.items) + inserted
            datasets.insert(
                insert_at,
                {
                    "label": f"{data_column} · SOL" if include_metric else "SOL",
                    "data": sol_data,
                    "is_sol": True,
                },
            )
            inserted += 1
        return labels, datasets

    def _build_chart_v2(self, section: ComparisonSection, chart_idx: int) -> dict[str, Any]:
        chart = super()._build_chart_v2(section, chart_idx)
        chart["sol_color"] = self.SOL_REFERENCE_COLOR
        return chart

    def _build_table_v2(self, section: ComparisonSection) -> dict[str, Any]:
        widest_df = max(section.dfs, key=len)
        show_diff = len(section.group.items) == 2
        data_headers = []
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
                data_headers.append({"name": "Difference", "differences_yaml": "", "metric": data_column})

        rows = []
        for row_idx in range(len(widest_df)):
            info_cells = [self._display_value(widest_df[column].get(row_idx, None)) for column in section.info_columns]
            data_cells = []
            for data_column in section.data_columns:
                raw_values = [df[data_column].get(row_idx, None) for df in section.dfs]
                show_sol = self._column_has_sol(section, data_column)
                for df, value in zip(section.dfs, raw_values, strict=True):
                    data_cells.append(self._display_value(value))
                    if show_sol:
                        sol = df[self._sol_column(data_column)].get(row_idx, None)
                        attainment = df[self._attainment_column(data_column)].get(row_idx, None)
                        data_cells.extend(
                            [
                                self._display_value(sol),
                                f"{attainment:.1%}" if self._numeric_value(attainment) is not None else "n/a",
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
