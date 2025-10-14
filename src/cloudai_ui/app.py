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

"""CloudAI UI — Dash.plotly based interactive visualization."""

from pathlib import Path

import dash
from dash import Input, Output, ctx, dcc, html

from .data_layer import LocalFileDataProvider
from .nccl_dashboard import NCCLDashboard


def available_dashboards() -> list[str]:
    """Determine available dashboard types based on data availability."""
    return ["nccl"]


def create_header_navbar(current_page: str, available_dashboards: list[str]):
    """Create the header with NVIDIA logo and navigation bar in the same row."""
    logo_title = html.Div(
        [
            html.Img(src="/assets/nvidia-logo.svg", className="nvidia-logo"),
            html.H1("CloudAI Dashboard", className="app-title"),
        ],
        className="header-left",
    )

    nav_bar = html.Nav(
        [
            dcc.Link("Home", href="/", className=f"nav-item {'active' if current_page == 'home' else ''}"),
            *[
                dcc.Link(
                    dashboard.title(),
                    href=f"/{dashboard}",
                    className=f"nav-item {'active' if current_page == dashboard else ''}",
                )
                for dashboard in available_dashboards
            ],
        ],
        className="header-nav",
    )

    return html.Div(
        [html.Div([logo_title, nav_bar], className="header-content")],
        className="header-container",
    )


def create_app(results_root: Path):
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, assets_folder="assets", title="CloudAI Dashboard", suppress_callback_exceptions=True)
    data_provider = LocalFileDataProvider(results_root)

    # Create stateful dashboard instances
    nccl_dashboard = NCCLDashboard(data_provider)

    app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname: str | None):
        """Route pages based on URL pathname."""
        dashboards = available_dashboards()

        if pathname == "/" or pathname is None:
            return create_main_page(dashboards, results_root)
        elif pathname == "/nccl":
            return html.Div([create_header_navbar("nccl", dashboards), nccl_dashboard.create_nccl_page()])
        else:
            return html.Div(
                [
                    html.H1("404 - Page Not Found"),
                    html.P(f"The page '{pathname}' was not found."),
                    html.A("Go Home", href="/"),
                ]
            )

    @app.callback(
        [Output("nccl-controls-container", "children"), Output("nccl-charts-container", "children")],
        [
            Input("nccl-time-range", "value"),
            Input("nccl-chart-controls", "value"),
            Input("nccl-system-filter", "value"),
            Input("nccl-scenario-filter", "value"),
            Input("nccl-group-by", "value"),
        ],
    )
    def update_nccl_dashboard(time_range_days, selected_charts, selected_systems, selected_scenarios, group_by):
        """Update NCCL dashboard."""
        return nccl_dashboard.update(
            ctx.triggered_id, time_range_days, selected_charts, selected_systems, selected_scenarios, group_by
        )

    return app


def create_main_page(dashboards: list[str], results_root: Path):
    """Create the main dashboard selection page."""
    dashboard_cards = html.Div([create_compact_dashboard_card(dashboard) for dashboard in dashboards])

    return html.Div(
        [
            create_header_navbar("home", dashboards),
            # Main content
            html.Div(
                [
                    html.Div(
                        [html.Strong("Results Directory: "), html.Code(str(results_root))],
                        className="dashboard-section",
                    ),
                    html.H3("Available Dashboards"),
                    dashboard_cards
                    if dashboards
                    else html.Div(
                        ["No dashboards available. No scenarios found in the results directory."],
                        className="dashboard-section",
                    ),
                ],
                className="main-content",
            ),
        ]
    )


def create_compact_dashboard_card(dashboard: str):
    """Create a compact dashboard card component."""
    description = {
        "nccl": " • NCCL performance analysis",
    }.get(dashboard, f" • {dashboard.title()} dashboard")

    return html.Div(
        [
            dcc.Link(
                [html.Strong(dashboard.title()), html.Span(description, className="text-muted")],
                href=f"/{dashboard}",
                className="compact-dashboard-link",
            ),
        ],
        className="compact-dashboard-card",
    )
