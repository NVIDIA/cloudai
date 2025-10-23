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
from typing import Any

import dash
from dash import Input, Output, ctx, dcc, html

from .data_layer import DataProvider, DataQuery, LocalFileDataProvider
from .dse_dashboard import DSEDashboard
from .nccl_dashboard import NCCLDashboard
from .nixl_dashboard import NIXLDashboard

APP_TITLE = "CloudAI UI (⍺)"


def available_dashboards() -> list[str]:
    """Determine available dashboard types based on data availability."""
    return ["nccl", "nixl", "dse"]


def create_header_navbar(current_page: str, available_dashboards: list[str]):
    """Create the header with NVIDIA logo and navigation bar in the same row."""
    logo_title = html.Div(
        [
            html.Img(src="/assets/nvidia-logo.svg", className="nvidia-logo"),
            html.H1(APP_TITLE, className="app-title"),
        ],
        className="header-left",
    )

    nav_bar = html.Nav(
        [
            dcc.Link("Home", href="/", className=f"nav-item {'active' if current_page == 'home' else ''}"),
            *[
                dcc.Link(
                    dashboard.upper(),
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
    app = dash.Dash(__name__, assets_folder="assets", title=APP_TITLE, suppress_callback_exceptions=True)
    data_provider = LocalFileDataProvider(results_root)

    # Create stateful dashboard instances
    nccl_dashboard = NCCLDashboard(data_provider, id_prefix="nccl")
    nixl_dashboard = NIXLDashboard(data_provider, id_prefix="nixl")
    dse_dashboard = DSEDashboard(data_provider, id_prefix="dse")

    app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname: str | None):
        """Route pages based on URL pathname."""
        dashboards = available_dashboards()

        if pathname == "/" or pathname is None:
            return create_main_page(dashboards, data_provider)
        elif pathname == "/nccl":
            return html.Div([create_header_navbar("nccl", dashboards), nccl_dashboard.create_page()])
        elif pathname == "/nixl":
            return html.Div([create_header_navbar("nixl", dashboards), nixl_dashboard.create_page()])
        elif pathname == "/dse":
            return html.Div([create_header_navbar("dse", dashboards), dse_dashboard.create_page()])
        else:
            return html.Div(
                [
                    html.H1("404 - Page Not Found"),
                    html.P(f"The page '{pathname}' was not found."),
                    html.A("Go Home", href="/"),
                ]
            )

    @app.callback(
        [
            Output(f"{nccl_dashboard.id_prefix}-controls-container", "children"),
            Output(f"{nccl_dashboard.id_prefix}-charts-container", "children"),
        ],
        [
            Input(f"{nccl_dashboard.id_prefix}-time-range", "value"),
            Input(f"{nccl_dashboard.id_prefix}-chart-controls", "value"),
            Input(f"{nccl_dashboard.id_prefix}-system-filter", "value"),
            Input(f"{nccl_dashboard.id_prefix}-scenario-filter", "value"),
            Input(f"{nccl_dashboard.id_prefix}-group-by", "value"),
        ],
    )
    def update_nccl_dashboard(time_range_days, selected_charts, selected_systems, selected_scenarios, group_by):
        """Update NCCL dashboard."""
        return nccl_dashboard.update(
            ctx.triggered_id, time_range_days, selected_charts, selected_systems, selected_scenarios, group_by
        )

    @app.callback(
        [
            Output(f"{nixl_dashboard.id_prefix}-controls-container", "children"),
            Output(f"{nixl_dashboard.id_prefix}-charts-container", "children"),
        ],
        [
            Input(f"{nixl_dashboard.id_prefix}-time-range", "value"),
            Input(f"{nixl_dashboard.id_prefix}-chart-controls", "value"),
            Input(f"{nixl_dashboard.id_prefix}-system-filter", "value"),
            Input(f"{nixl_dashboard.id_prefix}-scenario-filter", "value"),
            Input(f"{nixl_dashboard.id_prefix}-group-by", "value"),
        ],
    )
    def update_nixl_dashboard(time_range_days, selected_charts, selected_systems, selected_scenarios, group_by):
        """Update NIXL dashboard."""
        return nixl_dashboard.update(
            ctx.triggered_id, time_range_days, selected_charts, selected_systems, selected_scenarios, group_by
        )

    @app.callback(
        [
            Output(f"{dse_dashboard.id_prefix}-controls-container", "children"),
            Output(f"{dse_dashboard.id_prefix}-table-container", "children"),
        ],
        [
            Input(f"{dse_dashboard.id_prefix}-time-range", "value"),
            Input(f"{dse_dashboard.id_prefix}-system-filter", "value"),
            Input(f"{dse_dashboard.id_prefix}-scenario-filter", "value"),
            Input(f"{dse_dashboard.id_prefix}-run-selector", "value"),
        ],
    )
    def update_dse_dashboard(time_range_days, selected_systems, selected_scenarios, selected_dse_run):
        """Update DSE dashboard."""
        return dse_dashboard.update(
            ctx.triggered_id, time_range_days, selected_systems, selected_scenarios, selected_dse_run
        )

    return app


def create_main_page(dashboards: list[str], data_provider: DataProvider):
    """Create the main dashboard selection page."""
    dashboard_cards = html.Div([create_compact_dashboard_card(dashboard) for dashboard in dashboards])

    elements = [
        html.H3("Available Dashboards"),
        dashboard_cards
        if dashboards
        else html.Div(
            ["No dashboards available. No scenarios found in the results directory."],
            className="dashboard-section",
        ),
    ]

    if isinstance(data_provider, LocalFileDataProvider):
        data_provider.query_data(DataQuery(test_type=None, time_range_days=0))
        elements.extend(
            [
                html.H3("Data Provider Details"),
                html.Div(
                    [html.Strong("Results Directory: "), html.Code(str(data_provider.results_root))],
                    className="dashboard-section",
                ),
            ]
        )
        if data_provider.issues:
            issues_by_msg: dict[str, list[str]] = {}
            for issue in data_provider.issues:
                if issue.startswith("dir=") and ": " in issue:
                    dir_and_msg = issue.replace("dir=", "")
                    dir_path, msg = dir_and_msg.split(": ", 1)
                    issues_by_msg.setdefault(msg, []).append(dir_path)
                else:
                    issues_by_msg.setdefault("Other", []).append(issue)

            issue_sections: list[Any] = [html.H4("Warnings loading local data")]
            for msg, dirs in issues_by_msg.items():
                issue_sections.append(html.H5(msg, style={"marginTop": "1rem"}))
                issue_sections.append(
                    html.Ol(
                        [html.Li(html.Code(dir_path), className="alert alert-danger") for dir_path in dirs],
                        style={"paddingLeft": "2rem"},
                    )
                )

            elements.extend(issue_sections)

    return html.Div([create_header_navbar("home", dashboards), html.Div(elements, className="main-content")])


def create_compact_dashboard_card(dashboard: str):
    """Create a compact dashboard card component."""
    description = {
        "nccl": " • NCCL performance analysis",
        "nixl": " • NIXL benchmark performance",
        "dse": " • Design Space Exploration analysis",
    }.get(dashboard, f" • {dashboard.upper()} dashboard")

    return html.Div(
        [
            dcc.Link(
                [html.Strong(dashboard.upper()), html.Span(description, className="text-muted")],
                href=f"/{dashboard}",
                className="compact-dashboard-link",
            ),
        ],
        className="compact-dashboard-card",
    )
