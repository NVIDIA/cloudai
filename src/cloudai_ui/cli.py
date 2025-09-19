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

"""CloudAI Web UI CLI."""

import argparse
from pathlib import Path

from .app import create_app


def main():
    """Run the CloudAI Web UI Flask development server."""
    parser = argparse.ArgumentParser(description="CloudAI Web UI")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Path to results directory (default: results)"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Resolve results directory
    results_dir = Path(args.results_dir).resolve()

    app = create_app(str(results_dir))

    print("=" * 60)
    print("CloudAI Web UI")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down CloudAI Web UI...")


if __name__ == "__main__":
    main()
