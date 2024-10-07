# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from pathlib import Path
from typing import Callable, Optional

from .handlers import (
    handle_dry_run_and_run,
    handle_generate_report,
    handle_install_and_uninstall,
    handle_verify_systems,
    handle_verify_tests,
)


class CloudAICLI:
    """Command-line argument parser for Cloud AI and derivatives."""

    def __init__(self):
        self.DEFAULT_MODES = {
            "dry-run",
            "generate-report",
            "install",
            "run",
            "uninstall",
            "verify-systems",
            "verify-tests",
        }

        self.parser = argparse.ArgumentParser(description="Cloud AI")
        self.parser.add_argument(
            "--log-file", default="debug.log", help="The name of the log file (default: %(default)s)."
        )
        self.parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level (default: %(default)s).",
        )
        self.subparsers = self.parser.add_subparsers(dest="mode", required=True, title="modes")

        self.handlers: dict[str, Callable[[argparse.Namespace], int]] = {}

    def add_command(
        self,
        name: str,
        help_text: str,
        handler: Callable[[argparse.Namespace], int],
        system_config: Optional[bool] = None,
        test_templates_dir: Optional[bool] = None,
        tests_dir: Optional[bool] = None,
        test_scenario: Optional[bool] = None,
        output_dir: Optional[bool] = None,
    ) -> argparse.ArgumentParser:
        p = self.subparsers.add_parser(name, help=help_text)
        self.handlers[name] = handler
        if system_config is not None:
            p.add_argument(
                "--system-config", help="Path to the system configuration file.", required=system_config, type=Path
            )
        if test_templates_dir is not None:
            p.add_argument(
                "--test-templates-dir",
                help="Path to the test template configuration directory.",
                required=test_templates_dir,
                type=Path,
            )
        if tests_dir is not None:
            p.add_argument(
                "--tests-dir", help="Path to the test configuration directory.", required=tests_dir, type=Path
            )
        if test_scenario is not None:
            p.add_argument("--test-scenario", help="Path to the test scenario file.", required=test_scenario, type=Path)
        if output_dir is not None:
            p.add_argument("--output-dir", help="Path to the output directory.", required=output_dir, type=Path)

        return p

    def init_default_args(self) -> argparse.ArgumentParser:
        for mode in {"install", "uninstall"}:
            if mode not in self.DEFAULT_MODES:
                continue

            desc = "Prepare execution by setting up env and dependencies for the tests to run."
            if mode == "uninstall":
                desc = "Remove the installed dependencies."
            self.add_command(
                mode,
                desc,
                handle_install_and_uninstall,
                system_config=True,
                test_templates_dir=True,
                tests_dir=True,
                test_scenario=None,
                output_dir=False,
            )

        for mode in {"run", "dry-run"}:
            if mode not in self.DEFAULT_MODES:
                continue

            desc = "Execute the test scenarios."
            if mode == "dry-run":
                desc = "Perform a dry-run of the test scenarios without executing them."
            self.add_command(
                mode,
                desc,
                handle_dry_run_and_run,
                system_config=True,
                test_templates_dir=True,
                tests_dir=True,
                test_scenario=True,
                output_dir=False,
            )

        if "generate-report" in self.DEFAULT_MODES:
            p = self.add_command(
                "generate-report",
                "Generate a report based on the test results.",
                handle_generate_report,
                system_config=True,
                test_templates_dir=True,
                tests_dir=True,
                test_scenario=True,
                output_dir=True,
            )

        if "verify-systems" in self.DEFAULT_MODES:
            p = self.add_command("verify-systems", "Verify the system configurations.", handle_verify_systems)
            p.add_argument("system_configs", help="Path to the system configuration file or directory.", type=Path)

        if "verify-tests" in self.DEFAULT_MODES:
            p = self.add_command("verify-tests", "Verify the test configurations.", handle_verify_tests)
            p.add_argument("test_configs", help="Path to the test configuration file or directory.", type=Path)

        return self.parser

    def run(self) -> int:
        args = self.parser.parse_args()
        return self.handlers[args.mode](args)
