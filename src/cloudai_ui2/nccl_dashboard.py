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

"""NCCL Dashboard module for CloudAI UI v2."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from cloudai.core import TestRun
from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data

from .data_layer import DataProvider, DataQuery


class NCCLDashboard:
    """Stateful NCCL dashboard that caches loaded data."""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self._nccl_data: list[dict] | None = None
        self._time_range_days: int = 30  # Default to 30 days

    def load_data(self, time_range_days: int | None = None) -> None:
        """Load NCCL data from provider (cached after first call)."""
        if time_range_days is not None:
            self._time_range_days = time_range_days
            self._nccl_data = None  # Clear cache if time range changed

        if self._nccl_data is None:
            self._nccl_data = collect_nccl_data(self.data_provider, time_range_days=self._time_range_days)

    def get_data(self) -> list[dict]:
        """Get loaded NCCL data."""
        if self._nccl_data is None:
            self.load_data()
        return self._nccl_data or []

    def reload_data(self) -> None:
        """Force reload of data (e.g., on F5 or explicit refresh)."""
        self._nccl_data = None
        self.load_data()

    def render_controls(self) -> html.Div:
        """Render control elements."""
        data = self.get_data()
        return create_nccl_controls(data)

    def render_charts(
        self,
        selected_charts: list[str] | None = None,
        selected_scenarios: list[str] | None = None,
        time_range_days: int | None = None,
    ) -> Any:
        """Render charts with optional filters."""
        # Reload data if time range changed
        if time_range_days is not None and time_range_days != self._time_range_days:
            self.load_data(time_range_days)

        data = self.get_data()
        if not data:
            return html.Div(html.P("No NCCL test runs found.", className="text-muted"))

        # Use defaults if not specified
        if selected_charts is None:
            selected_charts = ["bandwidth_out", "bandwidth_in", "latency_out", "latency_in"]

        return _render_charts(data, selected_charts, selected_scenarios)

    def create_nccl_page(self) -> html.Div:
        """Create the NCCL page with initial data loaded."""
        return html.Div(
            [
                html.H3("NCCL Dashboard"),
                # Controls Section
                html.Div(self.render_controls(), id="nccl-controls-container"),
                # Charts Container
                html.Div(self.render_charts(), id="nccl-charts-container"),
            ],
            className="main-content",
        )


def create_scenario_dropdown_options(nccl_data: list[dict]) -> list[dict[str, Any]]:
    """Create dropdown options with unique scenario labels and their result counts."""
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


def extract_nccl_data_as_df(test_run: TestRun) -> pd.DataFrame:
    """Extract NCCL data as DataFrame, replicating NcclComparisonReport.extract_data_as_df logic."""
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


def collect_nccl_data(data_provider: DataProvider, time_range_days: int = 7) -> list[dict]:
    """Collect NCCL test runs and extract data using lazy loading."""
    query = DataQuery(test_type="nccl", time_range_days=time_range_days)
    scenarios = data_provider.query_data(query)

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


def create_nccl_bandwidth_chart(nccl_data: list[dict], selected_bandwidth_charts: list[str]) -> go.Figure:
    """Create bandwidth chart for NCCL test runs."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, data in enumerate(nccl_data):
        df = data["df"]
        label = data["label"]
        color = colors[i % len(colors)]

        if "bandwidth_out" in selected_bandwidth_charts:
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df["Busbw (GB/s) Out-of-place"],
                    mode="lines+markers",
                    name=f"{label} (Out-of-place)",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Size: %{x:,} B<br>"
                    + "Bandwidth: %{y:.2f} GB/s<br>"
                    + "<extra></extra>",
                )
            )

        if "bandwidth_in" in selected_bandwidth_charts:
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df["Busbw (GB/s) In-place"],
                    mode="lines+markers",
                    name=f"{label} (In-place)",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=6, symbol="square"),
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


def create_nccl_latency_chart(nccl_data: list[dict], selected_latency_charts: list[str]) -> go.Figure:
    """Create latency chart for NCCL test runs."""
    fig = go.Figure()

    # Use a simple color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i, data in enumerate(nccl_data):
        df = data["df"]
        label = data["label"]
        color = colors[i % len(colors)]

        if "latency_out" in selected_latency_charts:
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df["Time (us) Out-of-place"],
                    mode="lines+markers",
                    name=f"{label} (Out-of-place)",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Size: %{x:,} B<br>"
                    + "Latency: %{y:.2f} μs<br>"
                    + "<extra></extra>",
                )
            )

        if "latency_in" in selected_latency_charts:
            fig.add_trace(
                go.Scatter(
                    x=df["Size (B)"],
                    y=df["Time (us) In-place"],
                    mode="lines+markers",
                    name=f"{label} (In-place)",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=6, symbol="square"),
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


def create_nccl_controls(nccl_data: list[dict]) -> html.Div:
    """Create control elements for NCCL dashboard."""
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
                            {"label": "All time", "value": 36500},  # ~100 years
                        ],
                        value=30,  # Default to 30 days
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
                style={"marginBottom": "1rem", "display": "flex", "alignItems": "center"},
            ),
            # Scenario Filter
            html.Div(
                [
                    dcc.Dropdown(
                        options=create_scenario_dropdown_options(nccl_data),  # type: ignore
                        placeholder="Select Scenarios",
                        id="nccl-scenario-filter",
                        multi=True,
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


def _render_charts(nccl_data: list[dict], selected_charts: list[str] | None, selected_scenarios: list[str] | None):
    """Render charts based on data and filters."""
    # Filter by scenario if specified
    if selected_scenarios:
        nccl_data = [data for data in nccl_data if data["label"] in selected_scenarios]

    # Validate data and selections
    if not nccl_data or not selected_charts:
        if not nccl_data:
            message = (
                "No NCCL data available for the selected scenario." if selected_scenarios else "No NCCL data available."
            )
        else:
            message = "No charts selected."
        return html.Div(message, className="text-muted")

    charts = []

    # Check if any bandwidth charts are selected
    bandwidth_charts = [chart for chart in selected_charts if chart.startswith("bandwidth_")]
    if bandwidth_charts:
        bandwidth_fig = create_nccl_bandwidth_chart(nccl_data, bandwidth_charts)
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
        latency_fig = create_nccl_latency_chart(nccl_data, latency_charts)
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
