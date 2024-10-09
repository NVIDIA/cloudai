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
import asyncio
import logging
import logging.config
import sys
from pathlib import Path
from typing import List, Optional, Set

import toml

from cloudai import Parser, Registry, ReportGenerator, Runner, System, Test, TestParser, TestScenario, TestTemplate


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
            "verify-systems",
            "verify-tests",
            "verify-test-scenarios",
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
        template_type = type(test.test_template).__name__
        if template_type not in seen_names:
            seen_names.add(template_type)
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

    registry = Registry()
    installer_class = registry.installers_map.get(system.scheduler)
    if installer_class is None:
        raise NotImplementedError(f"No installer available for scheduler: {system.scheduler}")
    installer = installer_class(system)

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
    """
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    if mode == "run":
        logging.info("Checking if test templates are installed.")

        unique_templates = identify_unique_test_templates(tests)

        registry = Registry()
        installer_class = registry.installers_map.get(system.scheduler)
        if installer_class is None:
            raise NotImplementedError(f"No installer available for scheduler: {system.scheduler}")
        installer = installer_class(system)
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
    generator = ReportGenerator(output_dir)
    generator.generate_report(test_scenario)

    logging.info("Report generation completed.")


def init(
    system_config_path: Path, tests_dir: Path, test_scenario_path: Optional[Path]
) -> tuple[System, List[Test], Optional[TestScenario]]:
    parser = Parser(system_config_path)
    system, tests, test_scenario = parser.parse(tests_dir, test_scenario_path)
    return system, tests, test_scenario


def handle_verify_systems(root: Path) -> int:
    if not root.exists():
        logging.error(f"Tests directory {root} does not exist.")
        return 1

    test_tomls = [root]
    if root.is_dir():
        test_tomls = list(root.glob("*.toml"))
        if not test_tomls:
            logging.error(f"No test tomls found in {root}")
            return 1

    rc = 0
    for test_toml in test_tomls:
        logging.info(f"Verifying {test_toml}...")
        try:
            Parser.parse_system(test_toml)
        except Exception:
            rc = 1
            break
    if rc == 0:
        logging.info(f"Checked systems: {len(test_tomls)}, all passed")

    return rc


def handle_verify_tests(root: Path) -> int:
    if not root.exists():
        logging.error(f"Tests directory {root} does not exist.")
        return 1

    test_tomls = [root]
    if root.is_dir():
        test_tomls = list(root.glob("*.toml"))
        if not test_tomls:
            logging.error(f"No test tomls found in {root}")
            return 1

    rc = 0
    for test_toml in test_tomls:
        logging.info(f"Verifying {test_toml}...")
        try:
            parser = TestParser(Path(), None)  # type: ignore
            parser.current_file = test_toml
            parser.load_test_definition(toml.load(test_toml))
        except Exception:
            rc = 1
            break
    if rc == 0:
        logging.info(f"Checked tests: {len(test_tomls)}, all passed")

    return rc


def handle_verify_test_scenarios(root: Path, system_config: Path, tests_dir: Path) -> int:
    if not root.exists():
        logging.error(f"Tests directory {root} does not exist.")
        return 1

    test_tomls = [root]
    if root.is_dir():
        test_tomls = list(root.glob("*.toml"))
        if not test_tomls:
            logging.error(f"No test tomls found in {root}")
            return 1

    rc = 0
    for test_toml in test_tomls:
        logging.info(f"Verifying {test_toml}...")
        try:
            init(system_config, tests_dir, test_toml)
        except Exception:
            rc = 1
            break

    if rc == 0:
        logging.info(f"Checked scenarios: {len(test_tomls)}, all passed")

    return rc


def handle_runs_with_scenario(
    mode: str, output_dir: Optional[Path], system: System, tests: list[Test], test_scenario: TestScenario, log_file: str
) -> None:
    if mode in ["dry-run", "run"]:
        handle_dry_run_and_run(mode, system, tests, test_scenario)
        if mode == "run":
            logging.info(
                "All test scenario execution attempts are complete. Please review"
                f" the '{log_file}' file to confirm successful completion or to"
                " identify any issues."
            )
    elif mode == "generate-report":
        if not output_dir:
            logging.error("Error: --output-dir is required when mode is generate-report.")
            exit(1)
        handle_generate_report(test_scenario, output_dir)


def main() -> None:
    args = parse_arguments()

    setup_logging(args.log_file, args.log_level)

    if args.mode == "verify-systems":
        rc = handle_verify_systems(Path(args.system_config))
        exit(rc)
    elif args.mode == "verify-tests":
        rc = handle_verify_tests(Path(args.tests_dir))
        exit(rc)

    system_config_path = Path(args.system_config)
    tests_dir = Path(args.tests_dir)
    test_scenario_path = Path(args.test_scenario) if args.test_scenario else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.mode == "verify-test-scenarios":
        rc = handle_verify_test_scenarios(Path(args.test_scenario), system_config_path, tests_dir)
        exit(rc)

    logging.info(f"System configuration file: {system_config_path}")
    logging.info(f"Tests directory: {tests_dir}")
    logging.info(f"Test scenario file: {test_scenario_path}")
    logging.info(f"Output directory: {output_dir}")

    system, tests, test_scenario = init(system_config_path, tests_dir, test_scenario_path)
    if output_dir:
        system.output_path = output_dir.absolute()
    system.update()

    if args.mode in ["install", "uninstall"]:
        handle_install_and_uninstall(args.mode, system, tests)
    else:
        if not test_scenario:
            logging.error(f"Error: --test-scenario is required for mode={args.mode}")
            exit(1)
        handle_runs_with_scenario(args.mode, output_dir, system, tests, test_scenario, args.log_file)


if __name__ == "__main__":
    main()
