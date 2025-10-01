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

"""CloudAI UI v2 - Dash.plotly based interactive visualization."""

from pathlib import Path

import dash
from dash import Input, Output, dcc, html

from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition

from .data_layer import LocalFileDataProvider, TestScenarioInfo
from .nccl_dashboard import collect_nccl_data, create_nccl_page, update_nccl_charts


def available_dashboards(scenarios: list[TestScenarioInfo]) -> list[str]:
    """Determine available dashboard types based on test runs."""
    dash_types: list[str] = ["debug"]

    # Check for NCCL test runs
    nccl_data = collect_nccl_data(scenarios)
    if len(nccl_data) > 1:
        dash_types.append("nccl")

    return dash_types


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

    app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        """Route pages based on URL pathname."""
        scenarios = data_provider.get_scenarios()
        dashboards = available_dashboards(scenarios)

        if pathname == "/" or pathname is None:
            return create_main_page(scenarios, dashboards, results_root)
        elif pathname == "/debug":
            return create_debug_page(scenarios, dashboards, results_root)
        elif pathname == "/nccl":
            return create_nccl_page(scenarios, dashboards, results_root, create_header_navbar)
        else:
            return html.Div(
                [
                    html.H1("404 - Page Not Found"),
                    html.P(f"The page '{pathname}' was not found."),
                    html.A("Go Home", href="/"),
                ]
            )

    @app.callback(
        Output("nccl-charts-container", "children"),
        [Input("nccl-chart-controls", "value"), Input("nccl-scenario-filter", "value")],
        prevent_initial_call=False,
    )
    def update_nccl_charts_callback(selected_charts: list[str], selected_scenarios: list[str]):
        """Update NCCL charts based on user selection and scenario filter."""
        return update_nccl_charts(selected_charts, selected_scenarios, data_provider)

    return app


def create_main_page(scenarios: list[TestScenarioInfo], dashboards: list[str], results_root: Path):
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
        "debug": " • Debug view with scenario details",
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


def create_debug_page(scenarios: list[TestScenarioInfo], dashboards: list[str], results_root: Path):
    """Create the debug page with scenario categorization."""
    scenarios_with_runs = [s for s in scenarios if s.test_runs and not s.error]
    scenarios_empty = [s for s in scenarios if not s.test_runs and not s.error]
    scenarios_error = [s for s in scenarios if s.error]

    summary_div = html.Div(
        [
            html.H3("Debug Dashboard"),
            html.P(
                f"Total: {len(scenarios)} scenarios | With runs: {len(scenarios_with_runs)} | Empty: {len(scenarios_empty)} | Errors: {len(scenarios_error)}",
                className="text-muted",
            ),
            html.P(
                [html.Strong("Results Directory: "), html.Code(str(results_root))],
                className="text-muted",
            ),
        ],
        className="debug-summary",
    )

    scenarios_div = html.Div(
        ["No test scenarios found. Run some CloudAI tests to see results here."],
        className="dashboard-section",
    )
    if scenarios:
        scenarios_div = html.Div([create_expandable_scenario_item(scenario, i) for i, scenario in enumerate(scenarios)])

    return html.Div(
        [
            create_header_navbar("debug", dashboards),
            html.Div([summary_div, scenarios_div], className="main-content"),
        ]
    )


def create_scenario_section(title: str, scenarios: list[TestScenarioInfo], section_type: str = "default"):
    """Create a section for categorized scenarios."""
    if not scenarios:
        return None

    return html.Div(
        [
            html.H3(f"{title} ({len(scenarios)})"),
            html.Div([create_scenario_item(scenario, i, section_type) for i, scenario in enumerate(scenarios)]),
        ],
        className="dashboard-section",
    )


def create_scenario_item(scenario: TestScenarioInfo, index: int, section_type: str = "default"):
    """Create a scenario item component."""
    if scenario.error:
        # Error scenario
        return html.Div(
            [
                html.H4(scenario.name),
                html.P(scenario.timestamp.strftime("%Y-%m-%d %H:%M:%S"), className="text-muted"),
                html.Div([html.Strong("Error: "), scenario.error], className="text-error"),
            ],
            className="scenario-item scenario-error",
        )

    elif not scenario.test_runs:
        # Empty scenario
        return html.Div(
            [
                html.H4(scenario.name),
                html.P(scenario.timestamp.strftime("%Y-%m-%d %H:%M:%S"), className="text-muted"),
                html.P("0 runs", className="text-warning"),
            ],
            className="scenario-item scenario-empty",
        )

    else:
        # Scenario with test runs
        return html.Div(
            [
                html.H4(f"{scenario.name} ({len(scenario.test_runs)} runs)"),
                html.P(scenario.timestamp.strftime("%Y-%m-%d %H:%M:%S"), className="text-muted"),
                html.Div([create_test_run_item(test_run) for test_run in scenario.test_runs]),
            ],
            className="scenario-item scenario-success",
        )


def create_expandable_scenario_item(scenario: TestScenarioInfo, index: int):
    """Create an expandable scenario item with collapsible test runs."""

    if scenario.error:
        return html.Div(
            [
                html.Div(
                    [
                        html.Span("❌ ", className="status-icon"),
                        html.Strong(scenario.name),
                        html.Span(f" • {scenario.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", className="text-muted"),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Error: ", className="text-error"),
                        html.Span(scenario.error, className="text-muted"),
                    ]
                ),
            ],
            className="scenario-item scenario-error",
        )

    if not scenario.test_runs:
        return html.Div(
            [
                html.Span("⚠️ ", className="status-icon"),
                html.Strong(scenario.name),
                html.Span(f" • {scenario.timestamp.strftime('%Y-%m-%d %H:%M:%S')} • ", className="text-muted"),
                html.Span("0 runs", className="text-warning"),
            ],
            className="scenario-item scenario-empty",
        )

    return html.Details(
        [
            html.Summary(
                [
                    html.Span("✅ ", className="status-icon"),
                    html.Strong(f"{scenario.name} ({len(scenario.test_runs)} runs)"),
                    html.Span(f" • {scenario.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", className="text-muted"),
                ],
                className="scenario-summary",
            ),
            html.Div([create_test_run_item(test_run) for test_run in scenario.test_runs], className="test-runs-list"),
        ],
        className="scenario-item scenario-success expandable",
    )


def create_test_run_item(test_run):
    """Create a compact test run item component."""
    return html.Div(
        [
            html.Strong(test_run.name),
            html.Span(
                f" • {test_run.test.test_definition.test_template_name} • {test_run.iterations}x iterations",
                className="text-muted",
            ),
        ],
        className="compact-test-run-item",
    )
