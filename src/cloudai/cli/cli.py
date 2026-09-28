# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List
from unittest.mock import Mock

import click
import toml
import yaml

import cloudai.handlers
from cloudai.configurator.env_params import validate_domain_randomization_active
from cloudai.core import (
    JobSubmissionError,
    MissingTestError,
    Parser,
    Registry,
    System,
    SystemConfigParsingError,
    TestConfigParsingError,
    TestParser,
    TestScenarioParsingError,
)
from cloudai.parser import HOOK_ROOT
from cloudai.test_parser import load_test_toml_file
from cloudai.toml_utils import format_toml_decode_error


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


def _log_installation_dirs(prefix: str, system: System) -> None:
    logging.info(f"{prefix} '{system.install_path.absolute()}'. HF cache is {system.hf_home_path.absolute()}.")


def handle_install_and_uninstall(args: argparse.Namespace) -> int:
    """
    Manage the installation or uninstallation process for CloudAI.

    Based on user-specified mode, utilizing the Installer class.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config, args.hook_dir or HOOK_ROOT)
    system, tests, scenario = parser.parse(args.tests_dir, args.test_scenario)

    system.update()
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")

    installables, installer = cloudai.handlers.prepare_installation(system, tests, scenario)

    rc = 0
    if args.mode == "install":
        all_installed = installer.is_installed(installables)
        if all_installed:
            _log_installation_dirs("CloudAI is already installed into", system)
        else:
            logging.info("Not all components are ready")
            result = installer.install(installables)
            if result.success:
                _log_installation_dirs("CloudAI is successfully installed into", system)
            else:
                logging.error(result.message)
                rc = 1

    elif args.mode == "uninstall":
        logging.info("Uninstalling test templates.")
        result = installer.uninstall(installables)
        if result.success:
            logging.info("Uninstallation successful.")
        else:
            logging.error(result.message)
            rc = 1

    return rc


def register_signal_handlers(signal_handler: Callable) -> None:
    """Register signal handlers for handling termination-related signals."""
    signals = [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGQUIT,
    ]
    for sig in signals:
        signal.signal(sig, signal_handler)


def handle_generate_report(args: argparse.Namespace) -> int:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config, args.hook_dir or HOOK_ROOT)
    system, _, test_scenario = parser.parse(args.tests_dir, args.test_scenario)
    if test_scenario is None:
        raise ValueError("A test scenario is required.")

    cloudai.handlers.generate_reports(system, test_scenario, args.result_dir)

    logging.debug("Report generation completed.")

    return 0


def expand_file_list(root: Path, glob: str = "*.toml") -> tuple[int, List[Path]]:
    if not root.exists():
        logging.error(f"{root} does not exist.")
        return (1, [])

    test_tomls = [root]
    if root.is_dir():
        test_tomls = list(root.glob(glob))
        if not test_tomls:
            logging.error(f"No TOMLs found in {root}")
            return (1, [])

    return (0, test_tomls)


@contextmanager
def _ensure_kube_config_exists(system_toml_path: Path, content: str):
    try:
        config_dict = toml.loads(content)
    except toml.TomlDecodeError as e:
        logging.error(format_toml_decode_error(system_toml_path, e, "system config"))
        raise

    kube_config_path_str = config_dict.get("kube_config_path")
    kube_config_path = Path(kube_config_path_str) if kube_config_path_str else Path.home() / ".kube" / "config"

    created_file = False
    created_dir = False

    if not kube_config_path.exists():
        logging.warning(f"Kube config file '{kube_config_path}' not found. Creating a dummy one.")
        if not kube_config_path.parent.exists():
            kube_config_path.parent.mkdir(parents=True, exist_ok=True)
            created_dir = True

        dummy_config = {
            "apiVersion": "v1",
            "kind": "Config",
            "preferences": {},
            "clusters": [{"name": "dummy-cluster", "cluster": {"server": "https://dummy-server"}}],
            "users": [{"name": "dummy-user", "user": {"token": "dummy-token"}}],
            "contexts": [{"name": "dummy-context", "context": {"cluster": "dummy-cluster", "user": "dummy-user"}}],
            "current-context": "dummy-context",
        }
        kube_config_path.write_text(yaml.dump(dummy_config))
        created_file = True
    else:
        logging.debug(f"Kube config '{kube_config_path}' already exists. Skipping creation.")

    try:
        yield kube_config_path
    finally:
        if created_file:
            try:
                kube_config_path.unlink()
                logging.debug(f"Deleted temporary kube config: {kube_config_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary kube config '{kube_config_path}': {e}")
        if created_dir:
            try:
                kube_config_path.parent.rmdir()
                logging.debug(f"Deleted kube config directory: {kube_config_path.parent}")
            except OSError:
                pass


def verify_system_configs(system_tomls: List[Path]) -> int:
    nfailed = 0

    for system_toml in system_tomls:
        logging.debug(f"Verifying System: {system_toml}...")
        content = system_toml.read_text()

        if 'scheduler = "kubernetes"' in content:
            try:
                with _ensure_kube_config_exists(system_toml, content):
                    Parser.parse_system(system_toml)
            except Exception as e:
                logging.error(f"Failed to verify system config {system_toml}: {e}")
                logging.debug("", exc_info=True)
                nfailed += 1
        else:
            try:
                Parser.parse_system(system_toml)
            except Exception as e:
                logging.error(f"Failed to verify system config {system_toml}: {e}")
                logging.debug("", exc_info=True)
                nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(system_tomls)} system configurations have issues.")
    else:
        logging.info(f"Checked systems: {len(system_tomls)}, all passed")

    return nfailed


def verify_test_configs(test_tomls: List[Path]) -> int:
    nfailed = 0
    tp = TestParser([], None)  # type: ignore
    for test_toml in test_tomls:
        logging.debug(f"Verifying Test: {test_toml}...")
        try:
            with test_toml.open() as fh:
                tp.current_file = test_toml
                tp.load_test_definition(load_test_toml_file(fh, test_toml))
        except Exception as e:
            logging.error(f"Failed to verify Test: {test_toml}: {e}")
            logging.debug("", exc_info=True)
            nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(test_tomls)} test configurations have issues.")
    else:
        logging.info(f"Checked tests: {len(test_tomls)}, all passed")

    return nfailed


def verify_test_scenarios(
    scenario_tomls: List[Path], test_tomls: list[Path], hook_tomls: List[Path], hook_test_tomls: list[Path]
) -> int:
    system = Mock(spec=System, sol={})
    nfailed = 0
    for scenario_file in scenario_tomls:
        logging.debug(f"Verifying Test Scenario: {scenario_file}...")
        try:
            tests = Parser.parse_tests(test_tomls, system)
            hook_tests = Parser.parse_tests(hook_test_tomls, system)
            hooks = Parser.parse_hooks(hook_tomls, system, {t.name: t for t in hook_tests})
            scenario = Parser.parse_test_scenario(scenario_file, system, {t.name: t for t in tests}, hooks)
            validate_domain_randomization_active(scenario)
        except Exception as e:
            logging.error(f"Failed to verify Test Scenario: {scenario_file}: {e}")
            logging.debug("", exc_info=True)
            nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(scenario_tomls)} test scenarios have issues.")
    else:
        logging.info(f"Checked scenarios: {len(scenario_tomls)}, all passed")

    return nfailed


def handle_verify_all_configs(args: argparse.Namespace) -> int:
    root: Path = args.configs_dir
    err, tomls = expand_file_list(root, glob="**/*.toml")
    if err:
        return err

    err, hook_tomls = expand_file_list(HOOK_ROOT, glob="**/*.toml")
    tomls += hook_tomls
    logging.info(f"Found {len(hook_tomls)} hook TOMLs (always verified)")

    files = load_tomls_by_type(tomls)

    test_tomls = files["test"]
    if args.tests_dir:
        test_tomls = list(args.tests_dir.glob("*.toml"))
    elif files["scenario"]:
        logging.warning(
            "Test configuration directory not provided, using all found test TOMLs in the specified directory."
        )

    nfailed = 0
    total_checked = 0

    if files["system"]:
        nfailed += verify_system_configs(files["system"])
        total_checked += len(files["system"])
    if test_tomls:
        nfailed += verify_test_configs(test_tomls)
        total_checked += len(test_tomls)
    if files["scenario"]:
        nfailed += verify_test_scenarios(files["scenario"], test_tomls, files["hook"], files["hook_test"])
        total_checked += len(files["scenario"])
    if files["unknown"]:
        for unknown_file in files["unknown"]:
            logging.error(
                f"Unknown configuration file '{unknown_file}': could not classify as system, test, scenario, or hook."
            )
        nfailed += len(files["unknown"])
        total_checked += len(files["unknown"])

    if nfailed:
        logging.error(f"{nfailed} out of {total_checked} configuration files have issues.")
    else:
        logging.info(f"Checked {total_checked} configuration files, all passed")

    return nfailed


def load_tomls_by_type(tomls: List[Path]) -> dict[str, List[Path]]:
    files: dict[str, List[Path]] = {
        "system": [],
        "test": [],
        "scenario": [],
        "hook_test": [],
        "hook": [],
        "unknown": [],
    }
    for toml_file in tomls:
        content = toml_file.read_text()

        is_in_hook_root = False
        try:
            toml_file.relative_to(HOOK_ROOT)
            is_in_hook_root = True
        except ValueError:
            pass

        if is_in_hook_root:
            if "test" in toml_file.parts:
                files["hook_test"].append(toml_file)
            else:
                files["hook"].append(toml_file)
            continue

        try:
            toml_content = toml.loads(content)
        except toml.TomlDecodeError:
            files["unknown"].append(toml_file)
            continue

        if not isinstance(toml_content, dict):
            files["unknown"].append(toml_file)
            continue

        if "scheduler" in toml_content:
            files["system"].append(toml_file)
        elif "test_template_name =" in content and "[[Tests]]" not in content:
            files["test"].append(toml_file)
        elif "[[Tests]]" in content:
            files["scenario"].append(toml_file)
        else:
            files["unknown"].append(toml_file)

    return files


def handle_list_registered_items(item_type: str, verbose: bool) -> int:  # noqa: C901
    registry = Registry()
    if item_type.lower() == "reports":
        print("Available scenario reports:")
        for idx, (name, report) in enumerate(sorted(registry.scenario_reports.items()), start=1):
            string = f'{idx}. "{name}" {report.__name__}'
            if verbose:
                string += f" (config={registry.report_configs[name].model_dump_json(indent=None)})"
            print(string)
    elif item_type.lower() == "agents":
        print("Available agents:")
        for idx, name in enumerate(registry.agent_names(), start=1):
            agent = registry.get_agent(name)
            string = f'{idx}. "{name}" class={agent.__name__}'
            if verbose:
                string += f"{agent.__doc__}"
            print(string)
    elif item_type.lower() == "reward-functions":
        print("Available reward functions:")
        for idx, name in enumerate(registry.reward_function_names(), start=1):
            reward_function = registry.get_reward_function(name)
            callable_name = getattr(reward_function, "__name__", type(reward_function).__name__)
            string = f'{idx}. "{name}" function={callable_name}'
            if verbose:
                documentation = getattr(reward_function, "__doc__", None)
                if documentation:
                    string += f" {documentation}"
            print(string)

    return 0


def handle_dry_run_and_run(args: argparse.Namespace) -> int:
    try:
        system, tests, scenario = cloudai.handlers.load_scenario(
            args.test_scenario, args.system_config, tests_dir=args.tests_dir, hook_dir=args.hook_dir
        )
        if args.output_dir is not None:
            system.output_path = args.output_dir.absolute()
        runner = cloudai.handlers.create_experiment_runner(
            system, scenario, dry_run=args.mode == "dry-run", single_sbatch=args.single_sbatch
        )
        register_signal_handlers(runner.cancel_on_signal)
        logging.info("Results directory: %s", runner.runner.scenario_root)
        return cloudai.handlers.execute_experiment(runner, tests, args.enable_cache_without_check)
    except (
        MissingTestError,
        SystemConfigParsingError,
        TestConfigParsingError,
        TestScenarioParsingError,
        JobSubmissionError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        logging.error(str(exc))
        return 1


def common_options(f):
    f = click.option(
        "--system-config",
        "system_cfg",
        required=True,
        envvar="CLOUDAI_SYSTEM_CONFIG",
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        help="System config path. Can also be set via the CLOUDAI_SYSTEM_CONFIG environment variable.",
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
    f = click.option(
        "--hook-dir",
        required=False,
        default=None,
        type=click.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False, dir_okay=True),
        help="Directory with hook scenario and test configs.",
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
def install(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, hook_dir: Path | None):
    """Install the necessary components for workloads."""
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        hook_dir=hook_dir,
        mode="install",
    )
    exit(handle_install_and_uninstall(args))


@main.command()
@common_options
def uninstall(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, hook_dir: Path | None):
    """Uninstall the components used by workloads."""
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        hook_dir=hook_dir,
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
    hook_dir: Path | None,
    output_dir: Path,
    enable_cache_without_check: bool,
    single_sbatch: bool,
):
    """Dry run a scenario without executing it."""
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        hook_dir=hook_dir,
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
    hook_dir: Path | None,
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
        hook_dir=hook_dir,
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
def generate_report(system_cfg: Path, tests_dir: Path, scenario_cfg: Path, hook_dir: Path | None, result_dir: Path):
    """
    Generate a report from the results of a scenario.

    While this process is automatically executed as part of "run" command, one can also invoke it manually using this
    command.
    """
    args = argparse.Namespace(
        system_config=system_cfg,
        tests_dir=tests_dir,
        test_scenario=scenario_cfg,
        hook_dir=hook_dir,
        result_dir=result_dir,
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
    raise click.exceptions.Exit(1 if handle_verify_all_configs(args) else 0)


@main.command(name="list")
@click.argument("type", type=click.Choice(["reports", "agents", "reward-functions"], case_sensitive=False))
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
def list_registered(type: str, verbose: bool):
    """List available in Registry items."""
    handle_list_registered_items(type, verbose)
