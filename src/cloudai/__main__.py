# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
import logging
import os
import sys

from cloudai import Installer, Parser, ReportGenerator, Runner


def setup_logging(log_file: str, log_level: str) -> None:
    """
    Configure logging for the application.

    Args:
        log_level (str): The logging level (e.g., DEBUG, INFO).
        log_file (str): The name of the log file.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        filename=log_file,
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments, offering options for various operating modes, paths for installation, etc.

    Returns
        argparse.Namespace: An object containing all the parsed command-line
        arguments.
    """
    parser = argparse.ArgumentParser(description="Cloud AI")
    parser.add_argument(
        "--mode",
        default="run",
        choices=[
            "install",
            "dry-run",
            "run",
            "generate-report",
            "uninstall",
        ],
        help=(
            "Operating mode: 'install' to install test templates, 'dry-run' "
            "to simulate running experiments without checking for "
            "installation, 'run' to run experiments, 'generate-report' to "
            "generate a report from existing data, 'uninstall' to remove "
            "installed templates."
        ),
    )
    parser.add_argument(
        "--system_config_path",
        required=True,
        help="Path to the system configuration file.",
    )
    parser.add_argument(
        "--test_template_path",
        default="conf/v0.6/general/test_template",
        help="Path to the test template configuration directory.",
    )
    parser.add_argument(
        "--test_path",
        default="conf/v0.6/general/test",
        help="Path to the test configuration directory.",
    )
    parser.add_argument(
        "--test_scenario_path",
        required=False,
        help="Path to the test scenario file.",
    )
    parser.add_argument("--output_path", help="Path to the output directory.")
    parser.add_argument("--log_file", default="debug.log", help="The name of the log file.")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    return parser.parse_args()


def handle_install_and_uninstall(args: argparse.Namespace) -> None:
    """
    Manage the installation or uninstallation process for Cloud AI.

    Based on user-specified mode, utilizing the Installer and Parser classes.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
        user preferences.
    """
    logging.info("Starting configuration parsing")
    parser = Parser(args.system_config_path, args.test_template_path)
    system, test_templates = parser.parse_system_and_templates()

    if args.output_path:
        system.output_path = os.path.abspath(args.output_path)

    system.update()

    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")

    installer = Installer(system)

    if args.mode == "install":
        logging.info("Installing test templates.")
        if installer.is_installed(test_templates):
            print("Cloud AI is already installed.")
        else:
            result = installer.install(test_templates)
            if not result:
                print(result)
                sys.exit(1)

    elif args.mode == "uninstall":
        logging.info("Uninstalling test templates.")
        result = installer.uninstall(test_templates)
        if not result:
            print(result)
            sys.exit(1)


def handle_dry_run_and_run(args: argparse.Namespace) -> None:
    """
    Execute the dry-run or run modes for Cloud AI.

    Includes parsing configurations, verifying installations, and executing test scenarios.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
        user preferences.
    """
    logging.info("Starting configuration parsing")
    parser = Parser(
        args.system_config_path,
        args.test_template_path,
        args.test_path,
        args.test_scenario_path,
    )
    system, test_templates, test_scenario = parser.parse()

    if args.output_path:
        system.output_path = os.path.abspath(args.output_path)

    system.update()

    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    if args.mode == "run":
        logging.info("Checking if test templates are installed.")
        installer = Installer(system)
        result = installer.is_installed(test_templates)
        if not result:
            print("Cloud AI has not been installed. Please run install mode first.")
            print(result)
            sys.exit(1)

    test_scenario.pretty_print()

    runner = Runner(args.mode, system, test_scenario)
    asyncio.run(runner.run())

    print(f"All test scenario results stored at: {runner.runner.output_path}")

    if args.mode == "run":
        print(
            "All test scenario execution attempts are complete. Please review"
            " the 'debug.log' file to confirm successful completion or to"
            " identify any issues."
        )

        generator = ReportGenerator(runner.runner.output_path)
        generator.generate_report(test_scenario)


def handle_generate_report(args: argparse.Namespace) -> None:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
        user preferences.
    """
    logging.info("Generating report based on system and test templates")
    parser = Parser(
        args.system_config_path,
        args.test_template_path,
        args.test_path,
        args.test_scenario_path,
    )
    system, test_templates, test_scenario = parser.parse()

    generator = ReportGenerator(args.output_path)
    generator.generate_report(test_scenario)

    print("Report generation completed.")


def main() -> None:
    args = parse_arguments()

    setup_logging(args.log_file, args.log_level)

    if args.mode == "generate-report" and not args.output_path:
        print("Error: --output_path is required when mode is generate-report.")
        sys.exit(1)

    if args.mode in ["install", "uninstall"]:
        handle_install_and_uninstall(args)
    elif args.mode in ["dry-run", "run"]:
        handle_dry_run_and_run(args)
    elif args.mode == "generate-report":
        handle_generate_report(args)


if __name__ == "__main__":
    main()
