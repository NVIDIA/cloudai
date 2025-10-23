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

"""Base classes and generic functions for CloudAI dashboards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dash import dcc, html, no_update

from cloudai.report_generator.groups import GroupedItems, ItemsGrouper
from cloudai.report_generator.util import diff_test_runs

from .data_layer import DataProvider, DataQuery, Record


@dataclass(frozen=True)
class DashboardState:
    """Immutable state snapshot for rendering."""

    all_data: list[Record]
    filtered_data: list[Record]
    available_scenarios: list[str]  # Scenario labels available after system filter
    available_systems: list[str]  # System names available after scenario filter
    time_range_days: int
    selected_systems: list[str] | None
    selected_scenarios: list[str] | None
    selected_charts: list[str]
    group_by: list[str]


class ComparisonDashboard(ABC):
    """Abstract base class for comparison dashboards."""

    def __init__(self, data_provider: DataProvider, id_prefix: str):
        self.id_prefix = id_prefix
        self.data_manager = ComparisonDataManager(
            data_provider=data_provider,
            test_type=self.get_test_type(),
            default_charts=self.get_default_charts(),
            default_grouping=self.get_default_grouping(),
        )

    @abstractmethod
    def get_test_type(self) -> str:
        """Return the test type for data queries (e.g., 'nccl', 'ucc', 'nixl')."""
        pass

    @abstractmethod
    def get_chart_options(self) -> list[Any]:
        """Return chart type options for the dashboard."""
        pass

    @abstractmethod
    def get_default_charts(self) -> list[str]:
        """Return default selected charts."""
        pass

    @abstractmethod
    def get_default_grouping(self) -> list[str]:
        """Return default grouping fields."""
        pass

    @abstractmethod
    def get_page_title(self) -> str:
        """Return the dashboard page title."""
        pass

    @abstractmethod
    def render_charts_for_group(self, group: GroupedItems[Record], selected_charts: list[str]) -> list[Any]:
        """Render charts for a specific group. Returns list of Dash components."""
        pass

    def update_data(self, triggered_id: str, time_range_days: int | None) -> None:
        """Update data when time range changes."""
        if triggered_id == f"{self.id_prefix}-time-range":
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
        """Update dashboard based on user interactions."""
        if triggered_id is None:
            return (no_update, no_update)

        # time range affects everything else
        self.update_data(triggered_id, time_range_days)

        state = self.data_manager.get_state(selected_systems, selected_scenarios, selected_charts, group_by)

        # Update controls when time range, system filter, or scenario filter changes
        # (to refresh cascading dropdown options)
        if triggered_id in (
            f"{self.id_prefix}-time-range",
            f"{self.id_prefix}-system-filter",
            f"{self.id_prefix}-scenario-filter",
        ):
            return (
                render_controls(state, self.id_prefix, self.get_chart_options()),
                render_charts(state, self.render_charts_for_group),
            )

        return (no_update, render_charts(state, self.render_charts_for_group))

    def create_page(self) -> html.Div:
        """Create the dashboard page."""
        state = self.data_manager.get_state()
        return html.Div(
            [
                html.H3(self.get_page_title()),
                # Controls Section
                html.Div(
                    render_controls(state, self.id_prefix, self.get_chart_options()),
                    id=f"{self.id_prefix}-controls-container",
                ),
                # Charts Container
                html.Div(render_charts(state, self.render_charts_for_group), id=f"{self.id_prefix}-charts-container"),
            ],
            className="main-content",
        )


class ComparisonDataManager:
    """Base dashboard data manager."""

    def __init__(
        self,
        data_provider: DataProvider,
        test_type: str,
        default_charts: list[str],
        default_grouping: list[str],
    ):
        self.data_provider = data_provider
        self.test_type = test_type
        self.default_charts = default_charts
        self.default_grouping = default_grouping
        self._cached_data: list[Record] | None = None
        self.time_range_days: int = 30

    def load_data(self, time_range_days: int | None = None) -> None:
        if time_range_days is not None and time_range_days != self.time_range_days:
            self.time_range_days = time_range_days
            self._cached_data = None

        if self._cached_data is None:
            self._cached_data = [
                r
                for r in self.data_provider.query_data(
                    DataQuery(test_type=self.test_type, time_range_days=self.time_range_days)
                )
                if not r.df.empty
            ]

    def get_data(self) -> list[Record]:
        if self._cached_data is None:
            self.load_data()
        return self._cached_data or []

    def apply_filters(
        self, data: list[Record], selected_systems: list[str] | None, selected_scenarios: list[str] | None
    ) -> list[Record]:
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

        # Calculate data filtered only by systems (for scenario dropdown options)
        data_filtered_by_systems = self.apply_filters(all_data, selected_systems, None)
        # Calculate data filtered only by scenarios (for system dropdown options)
        data_filtered_by_scenarios = self.apply_filters(all_data, None, selected_scenarios)

        # Calculate final filtered data (both filters applied)
        filtered_data = self.apply_filters(all_data, selected_systems, selected_scenarios)

        if selected_charts is None:
            selected_charts = self.default_charts

        if group_by is None:
            group_by = self.default_grouping

        return DashboardState(
            all_data=all_data,
            filtered_data=filtered_data,
            available_scenarios=sorted({d.label for d in data_filtered_by_systems}),
            available_systems=sorted({d.system_name for d in data_filtered_by_scenarios}),
            time_range_days=self.time_range_days,
            selected_systems=selected_systems,
            selected_scenarios=selected_scenarios,
            selected_charts=selected_charts,
            group_by=group_by,
        )


# ============================================================================
# PRESENTATION LAYER
# ============================================================================


def get_grouping_options(data: list[Record]) -> list:
    """Get available grouping options from filtered data."""
    if not data:
        return []

    diff = diff_test_runs([record.test_run for record in data])
    options = [{"label": key, "value": key} for key in sorted(diff.keys())]
    return options


def render_controls(state: DashboardState, id_prefix: str, chart_options: list[Any]) -> html.Div:
    grouping_options = get_grouping_options(state.filtered_data)

    return html.Div(
        [
            # First row: Time Range, System Filter, Scenario Filter
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id=f"{id_prefix}-time-range",
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
                                placeholder="Time Range",
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=create_system_dropdown_options(state.available_systems),
                                placeholder="Filter Systems",
                                id=f"{id_prefix}-system-filter",
                                multi=True,
                                value=state.selected_systems,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=create_scenario_dropdown_options(state.available_scenarios, state.all_data),
                                placeholder="Filter Scenarios",
                                id=f"{id_prefix}-scenario-filter",
                                multi=True,
                                value=state.selected_scenarios,
                            ),
                        ],
                        style={"flex": "3"},
                    ),
                ],
                style={"display": "flex", "marginBottom": "1rem"},
            ),
            # Second row: Group By
            html.Div(
                [
                    html.Label("Group By", style={"fontWeight": "bold", "marginRight": "1rem", "minWidth": "80px"}),
                    dcc.Dropdown(
                        id=f"{id_prefix}-group-by",
                        options=grouping_options,
                        value=state.group_by,
                        placeholder="No grouping (all together)"
                        if grouping_options
                        else "No grouping options available",
                        multi=True,
                        disabled=not grouping_options,
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "1rem"},
            ),
            # Third row: Chart Types
            html.Div(
                [
                    html.Label("Chart types", style={"fontWeight": "bold", "marginRight": "1rem"}),
                    dcc.Checklist(
                        id=f"{id_prefix}-chart-controls",
                        options=chart_options,
                        value=state.selected_charts,
                        inline=True,
                        className="dash-checklist",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
        ],
        className="chart-controls",
    )


def format_group_title(group_name: str) -> Any:
    """Format group title with monospaced values (k=v format)."""
    if group_name == "all-in-one":
        return html.H3("All Results")

    # Split by spaces to get individual k=v pairs
    parts = group_name.split(" ")
    children = []

    for i, part in enumerate(parts):
        if i > 0:
            children.append(" ")  # Add space between parts

        if "=" in part:
            key, value = part.split("=", 1)
            children.append(key + "=")
            children.append(
                html.Code(
                    value, style={"backgroundColor": "#f5f5f5", "padding": "0.2rem 0.4rem", "borderRadius": "3px"}
                )
            )
        else:
            children.append(part)

    return html.H3(children)


def render_charts(state: DashboardState, render_charts_for_group_func: Any) -> Any:
    """Render charts using dashboard-specific rendering function."""
    if not state.filtered_data:
        return html.Div(html.P("No test runs found.", className="text-muted"))

    groups = ItemsGrouper[Record](items=state.filtered_data, group_by=state.group_by).groups()
    for grp in groups:
        for item in grp.items:
            item.name += f" @{item.item.system_name}"

    selected_charts = state.selected_charts
    charts = []

    for group in groups:
        group_content = []
        group_content.append(format_group_title(group.name))

        # Use dashboard-specific chart rendering
        chart_components = render_charts_for_group_func(group, selected_charts)
        group_content.extend(chart_components)

        # Wrap each group in a lightweight container
        charts.append(
            html.Div(
                group_content,
                style={
                    "borderLeft": "3px solid #76b900",  # NVIDIA green accent
                    "paddingLeft": "1.5rem",
                    "marginBottom": "2rem",
                    "marginTop": "1rem",
                },
            )
        )

    return charts


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_scenario_dropdown_options(scenario_labels: list[str], all_data: list[Record]) -> list:
    """Create scenario dropdown options with result counts."""
    # Count occurrences in all_data for each available scenario label
    label_counts: dict[str, int] = {}
    for data in all_data:
        if data.label in scenario_labels:
            label_counts[data.label] = label_counts.get(data.label, 0) + 1

    options = []
    for label in scenario_labels:
        count = label_counts.get(label, 0)
        result_text = "result" if count == 1 else "results"
        options.append({"label": f"{label} ({count} {result_text})", "value": label})

    return options


def create_system_dropdown_options(system_names: list[str]) -> list:
    """Create system dropdown options."""
    return [{"label": system, "value": system} for system in system_names]


def format_bytes(num_bytes: float) -> str:
    """Format byte size to human-readable format (KB, MB, GB, TB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            if num_bytes == int(num_bytes):
                return f"{int(num_bytes)}{unit}"
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def generate_size_ticks(x_values: set[float]) -> tuple[list[float], list[str]]:
    """
    Generate tick values and human-readable labels for size-based x-axis.

    Args:
        x_values: Set of x-axis values (in bytes)

    Returns:
        Tuple of (tick_values, tick_labels) for plotly
    """
    if not x_values:
        return ([], [])

    sorted_x = sorted(x_values)
    tick_values = sorted_x
    tick_labels = [format_bytes(x) for x in sorted_x]
    return (tick_values, tick_labels)
