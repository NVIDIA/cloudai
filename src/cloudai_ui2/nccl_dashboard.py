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

"""
NCCL Dashboard module for CloudAI UI v2.

Architecture:
- NCCLDataManager: Pure data management (loading, caching, filtering)
- DashboardState: Immutable state snapshot for rendering
- render_*(): Pure presentation functions (state -> UI)
- NCCLDashboard: Thin orchestration layer (coordinates data and rendering)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, no_update

from cloudai.core import TestRun
from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data

from .data_layer import DataProvider, DataQuery


@dataclass(frozen=True)
class DashboardState:
    """Immutable state snapshot for rendering."""

    all_data: list[dict]  # All loaded data
    filtered_data: list[dict]  # After applying system/scenario filters
    time_range_days: int
    selected_systems: list[str] | None
    selected_scenarios: list[str] | None
    selected_charts: list[str]


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
    ) -> tuple[Any, Any]:
        if triggered_id is None:
            return (no_update, no_update)

        # time range affects everything else
        self.update_data(triggered_id, time_range_days)

        state = self.data_manager.get_state(selected_systems, selected_scenarios, selected_charts)

        if triggered_id in ("nccl-time-range", "nccl-system-filter"):
            return (render_controls(state), render_charts(state))
        else:
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
        self._cached_data: list[dict] | None = None
        self.time_range_days: int = 30

    def load_data(self, time_range_days: int | None = None) -> None:
        if time_range_days is not None and time_range_days != self.time_range_days:
            self.time_range_days = time_range_days
            self._cached_data = None

        if self._cached_data is None:
            self._cached_data = self._collect_nccl_data()

    def get_data(self) -> list[dict]:
        if self._cached_data is None:
            self.load_data()
        return self._cached_data or []

    def apply_filters(
        self, data: list[dict], selected_systems: list[str] | None, selected_scenarios: list[str] | None
    ) -> list[dict]:
        if selected_systems:
            data = [d for d in data if cast(TestRun, d["test_run"]).test.test_template.system.name in selected_systems]

        if selected_scenarios:
            data = [d for d in data if d["label"] in selected_scenarios]

        return data

    def get_state(
        self,
        selected_systems: list[str] | None = None,
        selected_scenarios: list[str] | None = None,
        selected_charts: list[str] | None = None,
    ) -> DashboardState:
        all_data = self.get_data()
        filtered_data = self.apply_filters(all_data, selected_systems, selected_scenarios)

        if selected_charts is None:
            selected_charts = ["bandwidth_out", "bandwidth_in", "latency_out", "latency_in"]

        return DashboardState(
            all_data=all_data,
            filtered_data=filtered_data,
            time_range_days=self.time_range_days,
            selected_systems=selected_systems,
            selected_scenarios=selected_scenarios,
            selected_charts=selected_charts,
        )

    def _collect_nccl_data(self) -> list[dict]:
        query = DataQuery(test_type="nccl", time_range_days=self.time_range_days)
        scenarios = self.data_provider.query_data(query)

        nccl_data: list[dict] = []
        for scenario in scenarios:
            for test_run in scenario.test_runs:
                if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                    df = extract_nccl_data_as_df(test_run)
                    if not df.empty:
                        nccl_data.append(
                            {
                                "test_run": test_run,
                                "df": df,
                                "scenario_name": scenario.name,
                                "timestamp": scenario.timestamp,
                                "label": f"{scenario.name} - {scenario.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        )

        return nccl_data


# ============================================================================
# PRESENTATION LAYER
# ============================================================================


def render_controls(state: DashboardState) -> html.Div:
    all_data = state.all_data
    selected_systems = state.selected_systems
    time_range_days = state.time_range_days
    scenario_data = state.filtered_data
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
                        value=time_range_days,  # Use value from state
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
                                options=create_system_dropdown_options(all_data),
                                placeholder="Filter Systems",
                                id="nccl-system-filter",
                                multi=True,
                                value=selected_systems,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=create_scenario_dropdown_options(scenario_data),
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

    nccl_data = state.filtered_data
    selected_charts = state.selected_charts
    charts = []

    # Check if any bandwidth charts are selected
    bandwidth_charts = [chart for chart in selected_charts if chart.startswith("bandwidth_")]
    if bandwidth_charts:
        bandwidth_fig = create_bandwidth_chart(nccl_data, bandwidth_charts)
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
        latency_fig = create_latency_chart(nccl_data, latency_charts)
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


def create_scenario_dropdown_options(nccl_data: list[dict]) -> list:
    label_counts = {}
    for data in nccl_data:
        label = data["label"]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    options = []

    for label, count in label_counts.items():
        result_text = "result" if count == 1 else "results"
        options.append({"label": f"{label} ({count} {result_text})", "value": label})

    return options


def create_system_dropdown_options(nccl_data: list[dict]) -> list:
    systems = set()
    for data in nccl_data:
        tr = cast(TestRun, data["test_run"])
        systems.add(tr.test.test_template.system.name)

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


def create_bandwidth_chart(nccl_data: list[dict], selected_bandwidth_charts: list[str]) -> go.Figure:
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, data in enumerate(nccl_data):
        df = data["df"]
        label = data["label"]
        color = colors[i % len(colors)]

        for chart in selected_bandwidth_charts:
            fig_label = "In-place" if chart == "bandwidth_in" else "Out-of-place"
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df[f"Busbw (GB/s) {fig_label}"],
                    mode="lines+markers",
                    name=f"{label} ({fig_label})",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Size: %{x:,} B<br>"
                    + "Bandwidth: %{y:.2f} GB/s<br>"
                    + "<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis_title="Size (B)",
        yaxis_title="Bandwidth (GB/s)",
        xaxis_type="log",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode="zoom",
        height=600,
    )

    return fig


def create_latency_chart(nccl_data: list[dict], selected_latency_charts: list[str]) -> go.Figure:
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, data in enumerate(nccl_data):
        df = data["df"]
        label = data["label"]
        color = colors[i % len(colors)]

        for chart in selected_latency_charts:
            fig_label = "In-place" if chart == "latency_in" else "Out-of-place"
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df[f"Time (us) {fig_label}"],
                    mode="lines+markers",
                    name=f"{label} ({fig_label})",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Size: %{x:,} B<br>"
                    + "Latency: %{y:.2f} μs<br>"
                    + "<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis_title="Size (B)",
        yaxis_title="Time (μs)",
        xaxis_type="log",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode="zoom",
        height=600,
    )

    return fig
