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

from .data_layer import LocalFileDataProvider


def create_app(results_root: str | None = None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Use provided results_root or default to ./results
    if results_root is None:
        results_path = Path("results")
    else:
        results_path = Path(results_root)

    data_provider = LocalFileDataProvider(results_path)

    @app.route("/")
    def index():
        """Show all test scenarios."""
        scenarios = data_provider.get_scenarios()
        return render_template("index.html", scenarios=scenarios, results_root=results_path)

    return app
