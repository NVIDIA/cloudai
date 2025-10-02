# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, no_update

from cloudai.core import TestRun
from cloudai.report_generator.groups import GroupItem, ItemsGrouper
from cloudai.report_generator.util import diff_test_runs
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data

from .data_layer import DataProvider, DataQuery


@dataclass(frozen=True)
class NCCLRecord:
    """Immutable NCCL test run data."""

    test_run: TestRun
    df: pd.DataFrame
    scenario_name: str
    timestamp: datetime

    @property
    def label(self) -> str:
        return f"{self.scenario_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def system_name(self) -> str:
        return self.test_run.test.test_template.system.name


@dataclass(frozen=True)
class DashboardState:
    """Immutable state snapshot for rendering."""

    all_data: list[NCCLRecord]
    filtered_data: list[NCCLRecord]
    time_range_days: int
    selected_systems: list[str] | None
    selected_scenarios: list[str] | None
    selected_charts: list[str]
    group_by: list[str]


class NCCLDashboard:
    """Orchestrates data and presentation - thin coordination layer."""

    def __init__(self, data_provider: DataProvider):
        self.data_manager = NCCLDataManager(data_provider)

    def update_data(self, triggered_id: str, time_range_days: int | None) -> None:
        if triggered_id == "nccl-time-range":
            self.data_manager.load_data(time_range_days)

    def update(
        self,
        triggered_id: str | None,
        time_range_days: int | None,
        selected_charts: list[str] | None,
        selected_systems: list[str] | None,
        selected_scenarios: list[str] | None,
        group_by: list[str] | None,
    ) -> tuple[Any, Any]:
        if triggered_id is None:
            return (no_update, no_update)

        # time range affects everything else
        self.update_data(triggered_id, time_range_days)

        state = self.data_manager.get_state(selected_systems, selected_scenarios, selected_charts, group_by)

        if triggered_id == "nccl-time-range":
            return (render_controls(state), render_charts(state))

        return (no_update, render_charts(state))

    def create_nccl_page(self) -> html.Div:
        state = self.data_manager.get_state()
        return html.Div(
            [
                html.H3("NCCL Dashboard"),
                # Controls Section
                html.Div(render_controls(state), id="nccl-controls-container"),
                # Charts Container
                html.Div(render_charts(state), id="nccl-charts-container"),
            ],
            className="main-content",
        )


class NCCLDataManager:
    """Dashboard data manager."""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self._cached_data: list[NCCLRecord] | None = None
        self.time_range_days: int = 30

    def load_data(self, time_range_days: int | None = None) -> None:
        if time_range_days is not None and time_range_days != self.time_range_days:
            self.time_range_days = time_range_days
            self._cached_data = None

        if self._cached_data is None:
            self._cached_data = self._collect_nccl_data()

    def get_data(self) -> list[NCCLRecord]:
        if self._cached_data is None:
            self.load_data()
        return self._cached_data or []

    def apply_filters(
        self, data: list[NCCLRecord], selected_systems: list[str] | None, selected_scenarios: list[str] | None
    ) -> list[NCCLRecord]:
        if selected_systems:
            data = [d for d in data if d.system_name in selected_systems]

        if selected_scenarios:
            data = [d for d in data if d.label in selected_scenarios]

        return data

    def get_state(
        self,
        selected_systems: list[str] | None = None,
        selected_scenarios: list[str] | None = None,
        selected_charts: list[str] | None = None,
        group_by: list[str] | None = None,
    ) -> DashboardState:
        all_data = self.get_data()
        filtered_data = self.apply_filters(all_data, selected_systems, selected_scenarios)

        if selected_charts is None:
            selected_charts = ["bandwidth_out", "bandwidth_in", "latency_out", "latency_in"]

        if group_by is None:
            group_by = ["subtest_name"]

        return DashboardState(
            all_data=all_data,
            filtered_data=filtered_data,
            time_range_days=self.time_range_days,
            selected_systems=selected_systems,
            selected_scenarios=selected_scenarios,
            selected_charts=selected_charts,
            group_by=group_by,
        )

    def _collect_nccl_data(self) -> list[NCCLRecord]:
        query = DataQuery(test_type="nccl", time_range_days=self.time_range_days)
        scenarios = self.data_provider.query_data(query)

        nccl_data: list[NCCLRecord] = []
        for scenario in scenarios:
            for test_run in scenario.test_runs:
                if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                    df = extract_nccl_data_as_df(test_run)
                    if not df.empty:
                        nccl_data.append(
                            NCCLRecord(
                                test_run=test_run,
                                df=df,
                                scenario_name=scenario.name,
                                timestamp=scenario.timestamp,
                            )
                        )

        return nccl_data


# ============================================================================
# PRESENTATION LAYER
# ============================================================================


def get_grouping_options(data: list[NCCLRecord]) -> list:
    """Get available grouping options from filtered data."""
    if not data:
        return []

    diff = diff_test_runs([record.test_run for record in data])
    options = [{"label": key, "value": key} for key in sorted(diff.keys())]
    return options


def render_controls(state: DashboardState) -> html.Div:
    grouping_options = get_grouping_options(state.filtered_data)

    return html.Div(
        [
            # Time Range Picker
            html.Div(
                [
                    html.Label("Time Range:", style={"fontWeight": "bold", "marginRight": "1rem"}),
                    dcc.Dropdown(
                        id="nccl-time-range",
                        options=[
                            {"label": "Last 7 days", "value": 7},
                            {"label": "Last 14 days", "value": 14},
                            {"label": "Last 30 days", "value": 30},
                            {"label": "Last 60 days", "value": 60},
                            {"label": "Last 90 days", "value": 90},
                            {"label": "All time", "value": 0},
                        ],
                        value=state.time_range_days,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
                style={"marginBottom": "1rem", "display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=create_system_dropdown_options(state.all_data),
                                placeholder="Filter Systems",
                                id="nccl-system-filter",
                                multi=True,
                                value=state.selected_systems,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=create_scenario_dropdown_options(state.filtered_data),
                                placeholder="Filter Scenarios",
                                id="nccl-scenario-filter",
                                multi=True,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "marginBottom": "1rem"},
            ),
            # Group By Control
            html.Div(
                [
                    html.Label("Group By:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                    dcc.Dropdown(
                        id="nccl-group-by",
                        options=grouping_options,
                        value=state.group_by,
                        placeholder="No grouping (all together)"
                        if grouping_options
                        else "No grouping options available",
                        multi=True,
                        disabled=not grouping_options,
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            # Chart Controls
            html.Div(
                [
                    html.Label("Chart Controls:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                    dcc.Checklist(
                        id="nccl-chart-controls",
                        options=[
                            {"label": " Bandwidth Out-of-place", "value": "bandwidth_out"},
                            {"label": " Bandwidth In-place", "value": "bandwidth_in"},
                            {"label": " Latency Out-of-place", "value": "latency_out"},
                            {"label": " Latency In-place", "value": "latency_in"},
                        ],
                        value=["bandwidth_out", "bandwidth_in", "latency_out", "latency_in"],
                        inline=True,
                        className="dash-checklist",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"},
                    ),
                ],
            ),
        ],
        className="chart-controls",
    )


def render_charts(state: DashboardState) -> Any:
    if not state.filtered_data:
        return html.Div(html.P("No NCCL test runs found.", className="text-muted"))

    groups = ItemsGrouper[NCCLRecord](items=state.filtered_data, group_by=state.group_by).groups()

    selected_charts = state.selected_charts
    charts = []

    for group in groups:
        charts.append(html.H3(group.name))
        # Check if any bandwidth charts are selected
        bandwidth_charts = [chart for chart in selected_charts if chart.startswith("bandwidth_")]
        if bandwidth_charts:
            bandwidth_fig = create_bandwidth_chart(group.items, bandwidth_charts)
            charts.extend(
                [
                    html.H4("Bandwidth Performance"),
                    dcc.Graph(
                        figure=bandwidth_fig,
                        config={
                            "scrollZoom": True,
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        },
                    ),
                ]
            )

        # Check if any latency charts are selected
        latency_charts = [chart for chart in selected_charts if chart.startswith("latency_")]
        if latency_charts:
            latency_fig = create_latency_chart(group.items, latency_charts)
            charts.extend(
                [
                    html.H4("Latency Performance"),
                    dcc.Graph(
                        figure=latency_fig,
                        config={
                            "scrollZoom": True,
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        },
                    ),
                ]
            )

    return charts


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_scenario_dropdown_options(nccl_data: list[NCCLRecord]) -> list:
    label_counts: dict[str, int] = {}
    for data in nccl_data:
        label = data.label
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    options = []

    for label, count in label_counts.items():
        result_text = "result" if count == 1 else "results"
        options.append({"label": f"{label} ({count} {result_text})", "value": label})

    return options


def create_system_dropdown_options(nccl_data: list[NCCLRecord]) -> list:
    systems = set([data.system_name for data in nccl_data])
    return [{"label": system, "value": system} for system in systems]


def extract_nccl_data_as_df(test_run: TestRun) -> pd.DataFrame:
    stdout_path = test_run.output_path / "stdout.txt"

    if not stdout_path.exists():
        return pd.DataFrame()

    parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = extract_nccl_data(stdout_path)
    if not parsed_data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        parsed_data_rows,
        columns=[
            "Size (B)",
            "Count",
            "Type",
            "Redop",
            "Root",
            "Time (us) Out-of-place",
            "Algbw (GB/s) Out-of-place",
            "Busbw (GB/s) Out-of-place",
            "#Wrong Out-of-place",
            "Time (us) In-place",
            "Algbw (GB/s) In-place",
            "Busbw (GB/s) In-place",
            "#Wrong In-place",
        ],
    )

    df["GPU Type"] = gpu_type
    df["Devices per Node"] = num_devices_per_node
    df["Ranks"] = num_ranks

    df["Size (B)"] = df["Size (B)"].astype(int)
    df["Time (us) Out-of-place"] = df["Time (us) Out-of-place"].astype(float).round(2)
    df["Time (us) In-place"] = df["Time (us) In-place"].astype(float).round(2)
    df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
    df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
    df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
    df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)

    return df


def create_nccl_chart(
    nccl_data: list[GroupItem[NCCLRecord]],
    selected_charts: list[str],
    chart_prefix: str,
    y_column_prefix: str,
    hover_unit: str,
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, data in enumerate(nccl_data):
        df = data.item.df
        label = data.item.label
        color = colors[i % len(colors)]

        for chart in selected_charts:
            fig_label = "In-place" if chart == f"{chart_prefix}_in" else "Out-of-place"
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df[f"{y_column_prefix} {fig_label}"],
                    mode="lines+markers",
                    name=f"{label} {data.name} ({fig_label})",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Size: %{x:,} B<br>"
                    + f"{chart_prefix.title()}: %{{y:.2f}} {hover_unit}<br>"
                    + "<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis_title="Size (B)",
        yaxis_title=f"{chart_prefix.title()} ({hover_unit})",
        xaxis_type="log",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode="zoom",
        height=600,
    )

    return fig


def create_bandwidth_chart(nccl_data: list[GroupItem[NCCLRecord]], selected_bandwidth_charts: list[str]) -> go.Figure:
    """Create bandwidth chart for NCCL test runs."""
    return create_nccl_chart(
        nccl_data=nccl_data,
        selected_charts=selected_bandwidth_charts,
        chart_prefix="bandwidth",
        y_column_prefix="Busbw (GB/s)",
        hover_unit="GB/s",
    )


def create_latency_chart(nccl_data: list[GroupItem[NCCLRecord]], selected_latency_charts: list[str]) -> go.Figure:
    """Create latency chart for NCCL test runs."""
    return create_nccl_chart(
        nccl_data=nccl_data,
        selected_charts=selected_latency_charts,
        chart_prefix="latency",
        y_column_prefix="Time (us)",
        hover_unit="μs",
    )
