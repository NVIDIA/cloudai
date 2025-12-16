# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import logging.config
from pathlib import Path

import click

from .handlers import (
    handle_dry_run_and_run,
    handle_generate_report,
    handle_install_and_uninstall,
    handle_list_registered_items,
    handle_verify_all_configs,
)


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
            "bokeh": {
                "handlers": ["debug_file"],
                "propagate": False,
            },
            "kubernetes": {
                "handlers": [],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def common_options(f):
    f = click.option(
        "--system-config",
        "system_cfg",
        required=True,
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        help="System config path.",
    )(f)
    f = click.option(
        "--tests-dir",
        required=False,
        type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False, dir_okay=True),
        help="Directory with Test configs.",
    )(f)
    f = click.option(
        "--test-scenario",
        "scenario_cfg",
        required=True,
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        help="Scenario config path.",
    )(f)
    return f


output_dir_opt = click.option(
    "--output-dir",
    default=None,
    required=False,
    type=click.Path(resolve_path=True, path_type=Path, writable=True, file_okay=False),
    help="Output directory",
)
cache_without_check_opt = click.option(
    "--enable-cache-without-check",
    is_flag=True,
    default=False,
    help="Enable cache without checking.",
)
single_sbatch_opt = click.option(
    "--single-sbatch", is_flag=True, default=False, help="Use single sbatch for all test runs (Slurm only)."
)


@click.group(name="CloudAI", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--log-file", default="debug.log", help="Log file path for storing verbose output.")
@click.option("--log-level", default="INFO", help="Log level for standard output.")
@click.version_option(prog_name="CloudAI")
def main(log_file, log_level):
    """CloudAI is a benchmark framework focused on grading Data Center scale AI systems."""
    setup_logging(log_file, log_level)


@main.command()
@common_options
def install(system_cfg: Path, tests_dir: Path, scenario_cfg: Path):
    """Install the necessary components for workloads."""
    args = argparse.Namespace(system_config=system_cfg, tests_dir=tests_dir, test_scenario=scenario_cfg, mode="install")
    exit(handle_install_and_uninstall(args))


@main.command()
@common_options
def uninstall(system_cfg: Path, tests_dir: Path, scenario_cfg: Path):
    """Uninstall the components used by workloads."""
    args = argparse.Namespace(
        system_config=system_cfg, tests_dir=tests_dir, test_scenario=scenario_cfg, mode="uninstall"
    )
    exit(handle_install_and_uninstall(args))


@main.command()
@common_options
@output_dir_opt
@cache_without_check_opt
@single_sbatch_opt
def dry_run(
    system_cfg: Path,
    tests_dir: Path,
    scenario_cfg: Path,
    output_dir: Path,
    enable_cache_without_check: bool,
    single_sbatch: bool,
):
    """Dry run a scenario without executing it."""
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        output_dir=output_dir,
        mode="dry-run",
        enable_cache_without_check=enable_cache_without_check,
        single_sbatch=single_sbatch,
    )
    exit(handle_dry_run_and_run(args))


@main.command()
@common_options
@output_dir_opt
@cache_without_check_opt
@single_sbatch_opt
def run(
    system_cfg: Path,
    tests_dir: Path,
    scenario_cfg: Path,
    output_dir: Path,
    enable_cache_without_check: bool,
    single_sbatch: bool,
):
    """
    Run all the workloads from a scenario.

    It includes installing necessary components, executing the scenario, and generating reports.
    """
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        output_dir=output_dir,
        mode="run",
        enable_cache_without_check=enable_cache_without_check,
        single_sbatch=single_sbatch,
    )
    exit(handle_dry_run_and_run(args))


@main.command()
@common_options
@click.option(
    "--result-dir",
    required=True,
    type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False),
    help="Path to a scenario results directory.",
)
def generate_report(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, result_dir: Path):
    """
    Generate a report from the results of a scenario.

    While this process is automatically executed as part of "run" command, one can also invoke it manually using this
    command.
    """
    args = argparse.Namespace(
        system_config=system_cfg, tests_dir=tests_dir, test_scenario=scenario_cfg, result_dir=result_dir
    )
    exit(handle_generate_report(args))


@main.command()
@click.argument(
    "configs_dir",
    type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=True, dir_okay=True),
)
@click.option(
    "--tests-dir",
    type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False, dir_okay=True),
    help="Directory with Test configs.",
)
def verify_configs(configs_dir: Path, tests_dir: Path):
    """Verify the configuration TOML files."""
    args = argparse.Namespace(configs_dir=configs_dir, tests_dir=tests_dir)
    handle_verify_all_configs(args)


@main.command()
@click.argument("type", type=click.Choice(["reports", "agents"], case_sensitive=False))
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
def list(type: str, verbose: bool):
    """List available in Registry items."""
    handle_list_registered_items(type, verbose)
