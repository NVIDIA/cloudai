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

import argparse
from pathlib import Path

import click

from cloudai.cli import setup_logging
from cloudai.cli.handlers import (
    handle_dry_run_and_run,
    handle_generate_report,
    handle_install_and_uninstall,
    handle_list_registered_items,
    handle_verify_all_configs,
)


def common_options(f):
    f = click.option(
        "--system-config",
        "system_cfg",
        required=True,
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        help="System config path",
    )(f)
    f = click.option(
        "--tests-dir",
        required=True,
        type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False, dir_okay=True),
        help="Directory with Test configs",
    )(f)
    f = click.option(
        "--test-scenario",
        "scenario_cfg",
        required=True,
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        help="Scenario config path",
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
@click.option("--log-file", default="debug.log", help="Log file path")
@click.option("--log-level", default="INFO", help="Log level")
@click.version_option(prog_name="CloudAI")
def main(log_file, log_level):
    setup_logging(log_file, log_level)


@main.command()
@common_options
@output_dir_opt
def install(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, output_dir: Path):
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        output_dir=output_dir,
        mode="install",
    )
    exit(handle_install_and_uninstall(args))


@main.command()
@common_options
@output_dir_opt
def uninstall(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, output_dir: Path):
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        output_dir=output_dir,
        mode="uninstall",
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
@click.option("--strict", is_flag=True, default=False, help="Enable strict mode.")
def verify_configs(configs_dir: Path, tests_dir: Path, strict: bool):
    args = argparse.Namespace(configs_dir=configs_dir, tests_dir=tests_dir, strict=strict)
    handle_verify_all_configs(args)


@main.command()
@click.argument("type", type=click.Choice(["reports"]))
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
def list(type: str, verbose: bool):
    args = argparse.Namespace(type=type, verbose=verbose)
    handle_list_registered_items(args)
