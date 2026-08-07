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

"""Comparison report support for workloads that expose metric observations."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import cloudai.core
import cloudai.metrics
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonSection
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class MetricComparisonReport(ComparisonReport, abc.ABC):
    """
    Abstract comparison reporter that builds sections from metric observations.

    Workload reporters select their metrics and dimensions by implementing
    ``build_sections`` and calling ``build_metric_section``.
    """

    def _assessments(self, tr: cloudai.core.TestRun) -> list[cloudai.metrics.MetricAssessment]:
        return cloudai.metrics.assess_test_run_metrics(self.system, tr)

    def build_metric_section(
        self,
        group: GroupedTestRuns,
        metric: cloudai.metrics.MetricDefinition,
        x_dimension: str,
    ) -> ComparisonSection | None:
        """Build one explicitly configured metric section for a comparison group."""
        frames = [self._assessment_frame(self._assessments(item.tr), metric) for item in group.items]
        if all(frame.empty for frame in frames):
            return None
        return self._build_curve_section(group, metric, x_dimension, frames)

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

    @staticmethod
    def _build_curve_section(
        group: GroupedTestRuns,
        metric: cloudai.metrics.MetricDefinition,
        x_dimension: str,
        frames: list[pd.DataFrame],
    ) -> ComparisonSection:
        """Build measured curves, splitting repeated x values by their other dimensions."""
        value_columns = {"measured", "sol", "attainment"}
        series = list(
            dict.fromkeys(
                column
                for frame in frames
                for column in frame.columns
                if column != x_dimension and column not in value_columns
            )
        )
        split_series = any(not frame.empty and frame[x_dimension].duplicated().any() for frame in frames)
        series_keys = (
            list(
                dict.fromkeys(
                    tuple(row)
                    for frame in frames
                    if not frame.empty
                    for row in frame[series].drop_duplicates().itertuples(index=False, name=None)
                )
            )
            if split_series
            else [()]
        )
        if not split_series:
            series = []
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
                    result[f"{data_column} SOL"] = None
                    result[f"{data_column} % SOL"] = None
                    continue
                points = frame
                for dimension, value in zip(series, series_key, strict=True):
                    points = points[points[dimension] == value]
                points = points.groupby(x_dimension, as_index=False).first().set_index(x_dimension)
                result[data_column] = result[x_column].map(points["measured"])
                result[f"{data_column} SOL"] = result[x_column].map(points["sol"])
                result[f"{data_column} % SOL"] = result[x_column].map(points["attainment"])
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
