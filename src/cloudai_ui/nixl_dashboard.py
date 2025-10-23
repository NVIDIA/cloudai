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

"""NIXL-specific dashboard implementation."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from cloudai.report_generator.groups import GroupedItems, GroupItem

from .comparison_dashboard import ComparisonDashboard, format_bytes, generate_size_ticks
from .data_layer import DataProvider, Record


class NIXLDashboard(ComparisonDashboard):
    """NIXL-specific dashboard implementation."""

    def __init__(self, data_provider: DataProvider, id_prefix: str = "nixl"):
        super().__init__(data_provider, id_prefix)

    def get_test_type(self) -> str:
        return "nixlbench"

    def get_chart_options(self) -> list[Any]:
        return [
            {"label": "Bandwidth", "value": "bandwidth"},
            {"label": "Latency", "value": "latency"},
        ]

    def get_default_charts(self) -> list[str]:
        return ["bandwidth"]

    def get_default_grouping(self) -> list[str]:
        return ["op_type"]

    def get_page_title(self) -> str:
        return "NIXL Benchmark Dashboard"

    def render_charts_for_group(self, group: GroupedItems[Record], selected_charts: list[str]) -> list[Any]:
        """Render NIXL-specific charts for a group."""
        group_content = []

        if "bandwidth" in selected_charts:
            bandwidth_fig = create_bandwidth_chart(group.items)
            group_content.extend(
                [
                    html.H4("Bandwidth"),
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

        if "latency" in selected_charts:
            latency_fig = create_latency_chart(group.items)
            group_content.extend(
                [
                    html.H4("Latency"),
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


# ============================================================================
# NIXL-SPECIFIC MOCK CHART FUNCTIONS
# ============================================================================


def create_bandwidth_chart(nixl_data: list[GroupItem[Record]]) -> go.Figure:
    """Create bandwidth chart for NIXL test runs."""
    return create_nixl_chart(
        nixl_data=nixl_data,
        chart_prefix="bandwidth",
        y_column_prefix="bw_gb_sec",
        hover_unit="GB/s",
    )


def create_latency_chart(nixl_data: list[GroupItem[Record]]) -> go.Figure:
    """Create latency chart for NIXL test runs."""
    return create_nixl_chart(
        nixl_data=nixl_data,
        chart_prefix="latency",
        y_column_prefix="avg_lat",
        hover_unit="μs",
    )


def create_nixl_chart(
    nixl_data: list[GroupItem[Record]], chart_prefix: str, y_column_prefix: str, hover_unit: str
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    # Collect all x values to determine tick positions
    all_x_values = set()
    for data in nixl_data:
        df = data.item.df
        all_x_values.update(df["block_size"].unique())

    for i, data in enumerate(nixl_data):
        df = data.item.df
        label = data.item.label
        color = colors[i % len(colors)]

        fig_label = chart_prefix.title()
        fig.add_trace(
            go.Scatter(
                x=df["block_size"],
                y=df[y_column_prefix],
                mode="lines+markers",
                name=f"{label} {data.name} ({fig_label})",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Size: %{text}<br>"
                + f"{chart_prefix.title()}: %{{y:.2f}} {hover_unit}<br>"
                + "<extra></extra>",
                text=[format_bytes(x) for x in df["block_size"]],
            )
        )

    # Generate custom tick values and labels
    tickvals, ticktext = generate_size_ticks(all_x_values)

    fig.update_layout(
        xaxis_title="Block Size",
        yaxis_title=f"{chart_prefix.title()} ({hover_unit})",
        xaxis_type="log",
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45,
        ),
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
