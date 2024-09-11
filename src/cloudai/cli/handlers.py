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
from pathlib import Path
from typing import List, Set

from cloudai import Installer, Parser, ReportGenerator, Runner, Test, TestTemplate


def handle_verify_systems(args: argparse.Namespace) -> int:
    root: Path = args.system_configs
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

    return rc


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


def handle_install_and_uninstall(args: argparse.Namespace) -> int:
    """
    Manage the installation or uninstallation process for CloudAI.

    Based on user-specified mode, utilizing the Installer class.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config_path, args.test_templates_dir)
    system, tests, _ = parser.parse(args.tests_dir, None)

    if args.output_dir:
        system.output_path = args.output_dir.absolute()
    system.update()
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")

    unique_test_templates = identify_unique_test_templates(tests)
    installer = Installer(system)

    rc = 0
    if args.mode == "install":
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
                rc = 1

    elif args.mode == "uninstall":
        logging.info("Uninstalling test templates.")
        result = installer.uninstall(list(unique_test_templates))
        if result.success:
            logging.info("Uninstallation successful.")
        else:
            logging.error(result.message)
            rc = 1

    return rc


def handle_dry_run_and_run(args: argparse.Namespace) -> int:
    """
    Execute the dry-run or run modes for CloudAI.

    Includes parsing configurations, verifying installations, and executing test scenarios.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config, args.test_templates_dir)
    system, tests, test_scenario = parser.parse(args.tests_dir, args.test_scenario)
    assert test_scenario is not None

    if args.output_dir:
        system.output_path = args.output_dir.absolute()
    system.update()

    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    rc = 0
    if args.mode == "run":
        logging.info("Checking if test templates are installed.")

        unique_templates = identify_unique_test_templates(tests)

        installer = Installer(system)
        result = installer.is_installed(unique_templates)

        if not result.success:
            logging.error("CloudAI has not been installed. Please run install mode first.")
            logging.error(result.message)
            return 1

    logging.info(test_scenario.pretty_print())

    runner = Runner(args.mode, system, test_scenario)
    asyncio.run(runner.run())

    logging.info(f"All test scenario results stored at: {runner.runner.output_path}")

    if args.mode == "run":
        generator = ReportGenerator(runner.runner.output_path)
        generator.generate_report(test_scenario)
        logging.info(
            "All test scenario execution attempts are complete. Please review"
            f" the '{args.log_file}' file to confirm successful completion or to"
            " identify any issues."
        )

    return rc


def handle_generate_report(args: argparse.Namespace) -> int:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    logging.info("Generating report based on system and test scenario")
    generator = ReportGenerator(args.output_dir)
    generator.generate_report(args.test_scenario)

    logging.info("Report generation completed.")

    return 0
