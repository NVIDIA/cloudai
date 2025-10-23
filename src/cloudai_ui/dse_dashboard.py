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

"""DSE (Design Space Exploration) dashboard implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, cast

from dash import dash_table, dcc, html

from cloudai.core import TestTemplateStrategy
from cloudai.report_generator.util import diff_test_runs

from .data_layer import DataProvider, DataQuery, Record


class DSEDashboard:
    """DSE Dashboard - standalone dashboard for Design Space Exploration analysis."""

    def __init__(self, data_provider: DataProvider, id_prefix: str = "dse"):
        self.id_prefix = id_prefix
        self.data_manager = DSEDataManager(data_provider)

    def update_data(self, triggered_id: str, time_range_days: int | None) -> None:
        """Update data when time range changes."""
        if triggered_id == f"{self.id_prefix}-time-range":
            self.data_manager.load_data(time_range_days)

    def update(
        self,
        triggered_id: str | None,
        time_range_days: int | None,
        selected_systems: list[str] | None,
        selected_scenarios: list[str] | None,
        selected_dse_run: str | None,
    ) -> tuple[Any, Any]:
        """Update dashboard based on user interactions."""
        if triggered_id:
            self.update_data(triggered_id, time_range_days)

        all_data = self.data_manager.get_data()
        filtered_data = self.data_manager.apply_filters(all_data, selected_systems, selected_scenarios)

        available_systems = sorted({d.system_name for d in all_data})
        available_scenarios = sorted({d.scenario_name for d in all_data})
        available_dse_runs = sorted(set(r.dse_id for r in filtered_data))

        if not available_dse_runs:
            selected_dse_run = None
        else:
            if (selected_dse_run and selected_dse_run not in available_dse_runs) or not selected_dse_run:
                selected_dse_run = available_dse_runs[0]

        controls = render_dse_controls(
            id_prefix=self.id_prefix,
            time_range_days=self.data_manager.time_range_days,
            available_systems=available_systems,
            available_scenarios=available_scenarios,
            available_dse_runs=available_dse_runs,
            selected_systems=selected_systems,
            selected_scenarios=selected_scenarios,
            selected_dse_run=selected_dse_run,
        )

        records = self.data_manager.get_records_for_dse_run(selected_dse_run)
        if not selected_dse_run or not records:
            return (controls, html.Div())

        return (controls, html.Div([render_agent_info(records), render_dse_table(records)], style={"padding": "1rem"}))

    def create_page(self) -> html.Div:
        """Create the DSE dashboard page."""
        initial_controls, initial_content = self.update(
            triggered_id=None,
            time_range_days=self.data_manager.time_range_days,
            selected_systems=None,
            selected_scenarios=None,
            selected_dse_run=None,
        )

        return html.Div(
            [
                html.H3("DSE Dashboard"),
                html.Div(initial_controls, id=f"{self.id_prefix}-controls-container"),
                html.Div(initial_content, id=f"{self.id_prefix}-table-container"),
            ],
            className="main-content",
        )


class DSEDataManager:
    """Data manager for DSE dashboard."""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self._cached_data: list[Record] | None = None
        self.time_range_days: int = 30

    def load_data(self, time_range_days: int | None = None) -> None:
        """Load all test types data and filter to keep only DSE runs (multiple steps)."""
        if time_range_days is not None and time_range_days != self.time_range_days:
            self.time_range_days = time_range_days
            self._cached_data = None

        if self._cached_data is None:
            all_records = self.data_provider.query_data(DataQuery(test_type=None, time_range_days=self.time_range_days))

            dse_groups: defaultdict[str, list[Record]] = defaultdict(list)
            for record in all_records:
                dse_groups[record.dse_id].append(record)

            self._cached_data = []
            for records in dse_groups.values():
                if len(records) > 1:
                    self._cached_data.extend(records)

    def get_data(self) -> list[Record]:
        """Get cached data."""
        if self._cached_data is None:
            self.load_data()
        return self._cached_data or []

    def apply_filters(
        self, data: list[Record], selected_systems: list[str] | None, selected_scenarios: list[str] | None
    ) -> list[Record]:
        """Apply system and scenario filters."""
        if selected_systems:
            data = [d for d in data if d.system_name in selected_systems]

        if selected_scenarios:
            data = [d for d in data if d.scenario_name in selected_scenarios]

        return data

    def get_records_for_dse_run(self, dse_run_id: str | None) -> list[Record]:
        """Get all records for a specific DSE run."""
        if not dse_run_id:
            return []

        all_data = self.get_data()
        matching_records: list[Record] = [record for record in all_data if record.dse_id == dse_run_id]
        return sorted(matching_records, key=lambda x: x.test_run.step if hasattr(x.test_run, "step") else 0)


# ============================================================================
# UI COMPONENTS
# ============================================================================


def render_dse_controls(
    id_prefix: str,
    time_range_days: int,
    available_systems: list[str],
    available_scenarios: list[str],
    available_dse_runs: Iterable[str],
    selected_systems: list[str] | None,
    selected_scenarios: list[str] | None,
    selected_dse_run: str | None,
) -> html.Div:
    """Render DSE dashboard controls."""
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
                                value=time_range_days,
                                clearable=False,
                                placeholder="Time Range",
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[{"label": sys, "value": sys} for sys in available_systems],
                                placeholder="Filter Systems",
                                id=f"{id_prefix}-system-filter",
                                multi=True,
                                value=selected_systems,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[{"label": scen, "value": scen} for scen in available_scenarios],
                                placeholder="Filter Scenarios",
                                id=f"{id_prefix}-scenario-filter",
                                multi=True,
                                value=selected_scenarios,
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                ],
                style={"display": "flex", "marginBottom": "1rem"},
            ),
            # Second row: DSE Run selector
            html.Div(
                [
                    html.Label("DSE Run", style={"fontWeight": "bold", "marginRight": "1rem", "minWidth": "80px"}),
                    dcc.Dropdown(
                        id=f"{id_prefix}-run-selector",
                        options=[{"label": run, "value": run} for run in sorted(available_dse_runs)],
                        value=selected_dse_run,
                        placeholder="Select DSE Run" if available_dse_runs else "No DSE runs available",
                        clearable=False,
                        disabled=not available_dse_runs,
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "1rem"},
            ),
        ],
        className="chart-controls",
    )


def render_agent_info(records: list[Record]) -> html.Div:
    """Render agent information as a table."""
    tdef = records[0].test_run.test.test_definition

    table_data: list[dict] = [
        {
            "Name": tdef.agent,
            "Metrics": ", ".join(tdef.agent_metrics),
            "Reward Function": tdef.agent_reward_function,
            "Max Steps": str(tdef.agent_steps),
        }
    ]

    return html.Div(
        [
            html.H4("Agent Information"),
            dash_table.DataTable(
                data=table_data,
                columns=[
                    {"name": "Name", "id": "Name"},
                    {"name": "Metrics", "id": "Metrics"},
                    {"name": "Reward Function", "id": "Reward Function"},
                    {"name": "Max Steps", "id": "Max Steps"},
                ],
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "0.75rem",
                    "fontFamily": "'NVIDIA', sans-serif",
                    "fontSize": "0.95rem",
                },
                style_header={"backgroundColor": "#76b900", "color": "white", "fontWeight": "700", "textAlign": "left"},
                style_data={"backgroundColor": "#f7f7f7", "color": "#1a1a1a"},
                style_data_conditional=[  # type: ignore
                    {
                        "if": {"column_id": "Metrics"},
                        "fontFamily": "'RobotoMono', monospace",
                    },
                    {
                        "if": {"column_id": "Reward Function"},
                        "fontFamily": "'RobotoMono', monospace",
                    },
                ],
            ),
        ],
        style={"marginBottom": "1.5rem"},
    )


def render_dse_table(records: list[Record]) -> html.Div:
    """Render table showing steps for the DSE run."""
    diff = sorted(diff_test_runs([r.test_run for r in records]).keys(), key=lambda x: x.lower())
    table_data: list[dict] = []
    max_reward: float = 0.0

    for record in records:
        diff_fields: dict[str, str] = {}
        for fname in diff:
            if fname.startswith("extra_env_vars."):
                value = record.test_run.test.test_definition.extra_env_vars.get(fname.removeprefix("extra_env_vars."))
            else:
                value = TestTemplateStrategy._flatten_dict(record.test_run.test.test_definition.cmd_args_dict)[fname]
            diff_fields[fname] = str(value)

        reward, observation = "N/A", "N/A"
        if record.dse:
            reward = f"{record.dse.reward:.4f}"
            observation = ", ".join([f"{v:.2f}" for v in record.dse.observation])
            max_reward = max(max_reward, record.dse.reward)
        table_data.append({"Step": record.test_run.step, **diff_fields, "Reward": reward, "Observation": observation})

    style_data_conditional: list[dict[str, Any]] = []
    if max_reward:
        max_reward_str = f"{max_reward:.4f}"
        columns = list(table_data[0].keys())
        for col_idx, _ in enumerate(columns):
            style_config = {
                "if": {"filter_query": f"{{Reward}} = {max_reward_str}"},
                "borderTop": "3px solid #76b900",
                "borderBottom": "3px solid #76b900",
                "fontWeight": "600",
            }
            if col_idx == 0:
                style_config = {
                    "if": {"filter_query": f"{{Reward}} = {max_reward_str}", "column_id": columns[0]},
                    "borderLeft": "6px solid #76b900",
                }
            if col_idx == len(table_data[0].keys()) - 1:
                style_config = {
                    "if": {"filter_query": f"{{Reward}} = {max_reward_str}", "column_id": columns[-1]},
                    "borderRight": "3px solid #76b900",
                }
            style_data_conditional.append(style_config)

    return html.Div(
        [
            html.H4("Test Run Steps"),
            dash_table.DataTable(
                data=table_data,
                columns=[{"name": f.title(), "id": f} for f in table_data[0]],
                sort_action="native",
                sort_mode="multi",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "0.75rem",
                    "fontFamily": "'NVIDIA', Arial, Helvetica, sans-serif",
                    "fontSize": "0.95rem",
                },
                style_header={"backgroundColor": "#76b900", "color": "white", "fontWeight": "700", "textAlign": "left"},
                style_data={"backgroundColor": "#f7f7f7", "color": "#1a1a1a"},
                style_data_conditional=cast(Any, style_data_conditional),
            ),
        ],
        style={"marginTop": "1rem"},
    )
