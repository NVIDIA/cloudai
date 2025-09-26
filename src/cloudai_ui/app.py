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

"""CloudAI Web UI Flask application."""

from pathlib import Path

from flask import Flask, render_template

from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nixl_bench import NIXLBenchTestDefinition

from .data_layer import LocalFileDataProvider, TestScenarioInfo
from .nccl_dashboard import NCCLDashboard


def available_dashboards(scenarios: list[TestScenarioInfo]) -> list[str]:
    dash_types: list[str] = ["debug"]
    num_nccl, num_nixl = 0, 0
    for scenario in scenarios:
        for test_run in scenario.test_runs:
            if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                num_nccl += 1
            elif isinstance(test_run.test.test_definition, NIXLBenchTestDefinition):
                num_nixl += 1

    if num_nccl > 1:
        dash_types.append("nccl")
    if num_nixl > 1:
        dash_types.append("nixl")

    return dash_types


def create_app(results_root: Path):
    app = Flask(__name__)
    data_provider = LocalFileDataProvider(results_root)

    @app.route("/")
    def index():
        scenarios = data_provider.get_scenarios()
        dashboards = available_dashboards(scenarios)
        return render_template(
            "index.html",
            scenarios=scenarios,
            results_root=results_root,
            dashboards=dashboards,
            current_dashboard=None,  # Set to None to indicate we're on home page
        )

    @app.route("/<dashboard_type>")
    def dashboard(dashboard_type):
        scenarios = data_provider.get_scenarios()
        dashboards = available_dashboards(scenarios)

        if dashboard_type not in dashboards:
            return f"Dashboard '{dashboard_type}' not available", 404

        if dashboard_type == "debug":
            return render_template(
                "dashboards/debug.html",
                scenarios=scenarios,
                results_root=results_root,
                dashboards=dashboards,
                current_dashboard=dashboard_type,
            )
        elif dashboard_type == "nccl":
            # Generate NCCL dashboard using existing comparison report logic
            dashboard = NCCLDashboard(scenarios, results_root)
            dashboard_data = dashboard.get_dashboard_data()

            return render_template(
                "dashboards/nccl.html",
                results_root=results_root,
                dashboards=dashboards,
                current_dashboard=dashboard_type,
                **dashboard_data,
            )

        return f"Dashboard '{dashboard_type}' not implemented yet", 501

    return app
