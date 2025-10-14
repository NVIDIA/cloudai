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

"""NCCL-specific dashboard implementation."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from cloudai.report_generator.groups import GroupItem

from .comparisson_dashboard import ComparissonDashboard
from .data_layer import DataProvider, Record


class NCCLDashboard(ComparissonDashboard):
    """NCCL-specific dashboard implementation."""

    def __init__(self, data_provider: DataProvider, id_prefix: str = "nccl"):
        super().__init__(data_provider, id_prefix)

    def get_test_type(self) -> str:
        return "nccl"

    def get_chart_options(self) -> list[Any]:
        return [
            {"label": " BW out-of-place", "value": "bandwidth_out"},
            {"label": " BW in-place", "value": "bandwidth_in"},
            {"label": " Latency out-of-place", "value": "latency_out"},
            {"label": " Latency in-place", "value": "latency_in"},
        ]

    def get_default_charts(self) -> list[str]:
        return ["bandwidth_out"]

    def get_default_grouping(self) -> list[str]:
        return ["subtest_name"]

    def get_page_title(self) -> str:
        return "NCCL Dashboard"

    def render_charts_for_group(self, group: Any, selected_charts: list[str]) -> list[Any]:
        """Render NCCL-specific charts for a group."""
        group_content = []

        # Check if any bandwidth charts are selected
        bandwidth_charts = [chart for chart in selected_charts if chart.startswith("bandwidth_")]
        if bandwidth_charts:
            bandwidth_fig = create_bandwidth_chart(group.items, bandwidth_charts)
            group_content.extend(
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
            group_content.extend(
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

        return group_content


def create_bandwidth_chart(nccl_data: list[GroupItem[Record]], selected_bandwidth_charts: list[str]) -> go.Figure:
    """Create bandwidth chart for NCCL test runs."""
    return create_nccl_chart(
        nccl_data=nccl_data,
        selected_charts=selected_bandwidth_charts,
        chart_prefix="bandwidth",
        y_column_prefix="Busbw (GB/s)",
        hover_unit="GB/s",
    )


def create_latency_chart(nccl_data: list[GroupItem[Record]], selected_latency_charts: list[str]) -> go.Figure:
    """Create latency chart for NCCL test runs."""
    return create_nccl_chart(
        nccl_data=nccl_data,
        selected_charts=selected_latency_charts,
        chart_prefix="latency",
        y_column_prefix="Time (us)",
        hover_unit="μs",
    )


def create_nccl_chart(
    nccl_data: list[GroupItem[Record]],
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
        showlegend=True,
        legend=dict(
            orientation="v",  # Vertical orientation for better readability
            yanchor="top",
            y=-0.2,  # Position below the chart
            xanchor="left",
            x=0,  # Align to left
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        dragmode="zoom",
        height=600,
        margin=dict(b=150),  # More bottom margin for vertical legend
    )

    return fig
