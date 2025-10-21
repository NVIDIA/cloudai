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

"""CloudAI Web UI CLI - Dash.plotly based."""

from pathlib import Path

import click

from .app import create_app


@click.command()
@click.option(
    "--results-dir",
    default="results",
    help="Path to results directory",
    type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8050, help="Port to bind to", type=int)  # Dash default port
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(results_dir: Path, host: str, port: int, debug: bool):
    """Run the CloudAI Web UI development server."""
    app = create_app(results_dir)

    click.echo("=" * 60)
    click.echo("CloudAI Web UI ")
    click.echo("=" * 60)
    click.echo(f"Results directory: {results_dir}")
    click.echo(f"URL: http://{host}:{port}")
    click.echo()
    click.echo("Press Ctrl+C to stop the server")
    click.echo("=" * 60)

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        click.echo("\nShutting down CloudAI Web UI...")
