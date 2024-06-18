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
import logging.config
import sys
from pathlib import Path
from typing import List, Set

from cloudai import Installer, Parser, ReportGenerator, Runner, System, Test, TestScenario, TestTemplate


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

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
            "short": {"format": "[%(levelname)s] %(message)s"},
        },
        "handlers": {
            "default": {
                "level": log_level.upper(),
                "formatter": "short",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "debug_file": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": log_file,
                "mode": "w",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default", "debug_file"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments, offering options for various operating modes, paths for installation, etc.

    Returns
        argparse.Namespace: An object containing all the parsed command-line
        arguments.
    """
    parser = argparse.ArgumentParser(description="CloudAI")
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
        "--system-config",
        required=True,
        help="Path to the system configuration file.",
    )
    parser.add_argument(
        "--test-templates-dir",
        required=True,
        help="Path to the test template configuration directory.",
    )
    parser.add_argument(
        "--tests-dir",
        required=True,
        help="Path to the test configuration directory.",
    )
    parser.add_argument(
        "--test-scenario",
        required=False,
        help="Path to the test scenario file.",
    )
    parser.add_argument("--output-dir", help="Path to the output directory.")
    parser.add_argument("--log-file", default="debug.log", help="The name of the log file (default: %(default)s).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    return parser.parse_args()


def identify_unique_test_templates(tests: List[Test]) -> List[TestTemplate]:
    """
    Identify unique test templates from a list of tests.

    Args:
        tests (List[Test]): The list of test objects.

    Returns:
        List[TestTemplate]: The list of unique test templates.
    """
    unique_templates: List[TestTemplate] = []
    seen_names: Set[str] = set()

    for test in tests:
        template_name = test.test_template.name
        if template_name not in seen_names:
            seen_names.add(template_name)
            unique_templates.append(test.test_template)

    return unique_templates


def handle_install_and_uninstall(mode: str, system: System, tests: List[Test]) -> None:
    """
    Manage the installation or uninstallation process for CloudAI.

    Based on user-specified mode, utilizing the Installer class.

    Args:
        mode (str): The operating mode.
        system (System): The system object.
        tests (List[Test]): The list of test objects.
    """
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")

    unique_test_templates = identify_unique_test_templates(tests)
    installer = Installer(system)

    if mode == "install":
        all_installed = True
        for template in unique_test_templates:
            if not installer.is_installed([template]):
                all_installed = False
                logging.debug(f"Test template {template.name} is not installed.")
                break

        if all_installed:
            logging.info("CloudAI is already installed.")
        else:
            logging.info("Not all components are ready")
            result = installer.install(list(unique_test_templates))
            if result.success:
                logging.info("Installation successful.")
            else:
                logging.error(result.message)
                exit(1)

    elif mode == "uninstall":
        logging.info("Uninstalling test templates.")
        result = installer.uninstall(list(unique_test_templates))
        if result.success:
            logging.info("Uninstallation successful.")
        else:
            logging.error(result.message)
            sys.exit(1)


def handle_dry_run_and_run(mode: str, system: System, tests: List[Test], test_scenario: TestScenario) -> None:
    """
    Execute the dry-run or run modes for CloudAI.

    Includes parsing configurations, verifying installations, and executing test scenarios.

    Args:
        mode (str): The operating mode.
        system (System): The system object.
        tests (List[Test]): The list of test objects.
        test_scenario (TestScenario): The test scenario object.
        output_dir (Optional[Path]): The path to the output directory.
    """
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    if mode == "run":
        logging.info("Checking if test templates are installed.")

        unique_templates = identify_unique_test_templates(tests)

        installer = Installer(system)
        result = installer.is_installed(unique_templates)

        if not result.success:
            logging.error("CloudAI has not been installed. Please run install mode first.")
            logging.error(result.message)
            exit(1)

    logging.info(test_scenario.pretty_print())

    runner = Runner(mode, system, test_scenario)
    asyncio.run(runner.run())

    logging.info(f"All test scenario results stored at: {runner.runner.output_path}")

    if mode == "run":
        logging.info(
            "All test scenario execution attempts are complete. Please review"
            " the 'debug.log' file to confirm successful completion or to"
            " identify any issues."
        )

        generator = ReportGenerator(runner.runner.output_path)
        generator.generate_report(test_scenario)


def handle_generate_report(test_scenario: TestScenario, output_dir: Path) -> None:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        test_scenario (TestScenario): The test scenario object.
        output_dir (Path): The path to the output directory.
    """
    logging.info("Generating report based on system and test scenario")
    generator = ReportGenerator(str(output_dir))
    generator.generate_report(test_scenario)

    logging.info("Report generation completed.")


def main() -> None:
    args = parse_arguments()

    setup_logging(args.log_file, args.log_level)

    system_config_path = Path(args.system_config)
    test_templates_dir = Path(args.test_templates_dir)
    tests_dir = Path(args.tests_dir)
    test_scenario_path = Path(args.test_scenario) if args.test_scenario else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    logging.info(f"System configuration file: {system_config_path}")
    logging.info(f"Test templates directory: {test_templates_dir}")
    logging.info(f"Tests directory: {tests_dir}")
    logging.info(f"Test scenario file: {test_scenario_path}")
    logging.info(f"Output directory: {output_dir}")

    parser = Parser(system_config_path, test_templates_dir)
    system, tests, test_scenario = parser.parse(tests_dir, test_scenario_path)

    if output_dir:
        system.output_path = str(output_dir.absolute())
    system.update()

    if args.mode in ["install", "uninstall"]:
        handle_install_and_uninstall(args.mode, system, tests)
    else:
        if not test_scenario:
            logging.error(f"Error: --test-scenario is required for mode={args.mode}")
            exit(1)

        elif args.mode in ["dry-run", "run"]:
            handle_dry_run_and_run(args.mode, system, tests, test_scenario)
        elif args.mode == "generate-report":
            if not output_dir:
                logging.error("Error: --output-dir is required when mode is generate-report.")
                exit(1)
            handle_generate_report(test_scenario, output_dir)


if __name__ == "__main__":
    main()
