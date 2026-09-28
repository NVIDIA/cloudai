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
import copy
import logging
import signal
import traceback
from pathlib import Path
from typing import Callable, Optional

from cloudai.configurator.env_params import validate_domain_randomization_active
from cloudai.core import (
    BaseInstaller,
    CloudAIGymEnv,
    Installable,
    InstallStatusResult,
    MissingTestError,
    Parser,
    Registry,
    Runner,
    System,
    TestScenario,
    TestScenarioParsingError,
)
from cloudai.models.scenario import ReportConfig
from cloudai.models.workload import TestDefinition
from cloudai.parser import HOOK_ROOT
from cloudai.systems.slurm import SingleSbatchRunner, SlurmSystem
from cloudai.util import prepare_output_dir


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

    installables, installer = prepare_installation(system, tests, scenario)

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


def prepare_installation(
    system: System, tests: list[TestDefinition], scenario: Optional[TestScenario]
) -> tuple[list[Installable], BaseInstaller]:
    installables: list[Installable] = []
    if scenario:
        installables.extend(_scenario_installables(scenario))
    else:
        for test in tests:
            logging.debug(f"{test.name} has {len(test.installables)} installables.")
            installables.extend(test.installables)

    registry = Registry()
    installer_class = registry.installers_map.get(system.scheduler, BaseInstaller)
    if installer_class is None:
        raise NotImplementedError(f"No installer available for scheduler: {system.scheduler}")
    installer = installer_class(system)

    return installables, installer


def _scenario_installables(scenario: TestScenario) -> list[Installable]:
    installables: list[Installable] = []
    for test_run in scenario.test_runs:
        logging.debug(f"{test_run.test.name} has {len(test_run.test.installables)} installables.")
        installables.extend(test_run.test.installables)
        for hook in (test_run.pre_test, test_run.post_test):
            if hook is not None:
                installables.extend(_scenario_installables(hook))
    return installables


def handle_dse_job(runner: Runner, args: argparse.Namespace) -> int:
    registry = Registry()

    original_test_runs = copy.deepcopy(runner.runner.test_scenario.test_runs)

    has_dependencies = any(tr.dependencies for tr in runner.runner.test_scenario.test_runs)
    if has_dependencies:
        logging.error(
            "Dependencies are not supported for DSE jobs, all cases run consecutively. "
            "Please remove dependencies and re-run."
        )
        return 1

    err = 0
    # Capture an unexpected error so reports still generate, then re-raise below.
    run_error: Exception | None = None
    try:
        for tr in runner.runner.test_scenario.test_runs:
            test_run = copy.deepcopy(tr)

            agent_type = test_run.test.agent
            if not registry.has_agent(agent_type):
                logging.error(
                    f"No agent available for type: {agent_type}. Please make sure {agent_type} "
                    f"is a valid agent type. Available agents: {registry.agent_names()}"
                )
                err = 1
                continue
            agent_class = registry.get_agent(agent_type)

            agent_config_data = test_run.test.agent_config or {}
            agent_config = agent_class.get_config_class()(**agent_config_data)
            env = CloudAIGymEnv(
                test_run=test_run,
                runner=runner.runner,
                rewards=agent_config.rewards,
            )
            if agent_config.start_action == "first":
                logging.info(f"Using deterministic first sweep for the chosen agent: {env.first_sweep}.")

            agent = agent_class(env, agent_config)
            logging.debug(f"Created agent {agent.__class__.__name__}.")

            env.update_output()
            try:
                err |= agent.run()
            finally:
                env.update_output()
    except Exception as exc:
        run_error = exc
        logging.exception("DSE job aborted by an unexpected error; generating reports before failing.")

    if args.mode == "run":
        runner.runner.test_scenario.test_runs = original_test_runs
        generate_reports(
            runner.runner.system,
            runner.runner.test_scenario,
            runner.runner.scenario_root,
            error=run_error,
        )

    if run_error is not None:
        raise run_error.with_traceback(run_error.__traceback__)

    logging.info("All jobs are complete.")
    return err


def _record_run_failure(result_dir: Path, error: BaseException) -> None:
    """Persist an aborting error into the results dir so the failure is documented with the reports."""
    failure_path = result_dir / "dse_failure.txt"
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    try:
        result_dir.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(f"DSE job aborted by an unexpected {type(error).__name__}: {error}\n\n{tb}")
        logging.info(f"Documented the aborting error in {failure_path}")
    except OSError:
        logging.exception(f"Failed to write failure report to {failure_path}")


def generate_reports(
    system: System,
    test_scenario: TestScenario,
    result_dir: Path,
    error: BaseException | None = None,
) -> None:
    registry = Registry()

    if error is not None:
        _record_run_failure(result_dir, error)

    for name, reporter_class in registry.ordered_scenario_reports():
        logging.debug(f"Generating report '{name}' ({reporter_class.__name__})")

        cfg = registry.report_configs.get(name, ReportConfig(enable=False))
        if scenario_cfg := test_scenario.reports.get(name):
            cfg = scenario_cfg
        elif isinstance(system, SlurmSystem) and system.reports and name in system.reports:
            cfg = system.reports[name]
        logging.debug(f"Report '{name}' config is: {cfg.model_dump_json(indent=None)}")

        if not cfg.enable:
            logging.debug(f"Skipping report {name} because it is disabled.")
            continue

        try:
            reporter = reporter_class(system, test_scenario, result_dir, cfg)
            reporter.generate()
        except Exception as e:
            logging.warning(f"Error generating report '{name}', see debug log for details")
            logging.debug(e, exc_info=True)


def handle_non_dse_job(runner: Runner, args: argparse.Namespace) -> bool:
    successful = runner.run()
    generate_reports(runner.runner.system, runner.runner.test_scenario, runner.runner.scenario_root)
    logging.info("All jobs are complete.")
    return successful


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


def _setup_system_and_scenario(
    args: argparse.Namespace,
) -> tuple[System, TestScenario, list[TestDefinition]] | None:
    parser = Parser(args.system_config, args.hook_dir or HOOK_ROOT)
    try:
        system, tests, test_scenario = parser.parse(args.tests_dir, args.test_scenario)
    except MissingTestError as e:
        logging.error(e.message)
        return None

    assert test_scenario is not None

    if args.output_dir:
        system.output_path = args.output_dir.absolute()

    if not prepare_output_dir(system.output_path):
        return None

    if args.mode == "dry-run":
        system.monitor_interval = 1
    system.update()

    return system, test_scenario, tests


def _handle_single_sbatch(args: argparse.Namespace, system: System) -> bool:
    if not args.single_sbatch:
        return True

    if not isinstance(system, SlurmSystem):
        logging.error("Single sbatch is only supported for Slurm systems.")
        return False

    Registry().update_runner("slurm", SingleSbatchRunner)
    return True


def _check_installation(
    args: argparse.Namespace, system: System, tests: list[TestDefinition], test_scenario: TestScenario
) -> InstallStatusResult:
    logging.info("Checking if workloads components are installed.")
    installables, installer = prepare_installation(system, tests, test_scenario)

    if args.enable_cache_without_check:
        result = installer.mark_as_installed(installables)
    else:
        result = installer.is_installed(installables)

    return result


def handle_dry_run_and_run(args: argparse.Namespace) -> int:
    setup_result = _setup_system_and_scenario(args)
    if setup_result is None:
        return 1
    system, test_scenario, tests = setup_result

    try:
        validate_domain_randomization_active(test_scenario)
    except TestScenarioParsingError as e:
        logging.error(str(e))
        return 1

    if not _handle_single_sbatch(args, system):
        return 1

    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    result = _check_installation(args, system, tests, test_scenario)
    if args.mode == "run" and not result.success:
        logging.info("Not all workloads components are installed. Installing...")
        installables, installer = prepare_installation(system, tests, test_scenario)

        result = installer.install(installables)
        if result.success:
            _log_installation_dirs("CloudAI is successfully installed into", system)
        else:
            logging.error("Failed to install workloads components.")
            logging.error(result.message)
            return 1
    elif args.mode == "dry-run":
        # simulate installation for dry-run
        installables, installer = prepare_installation(system, tests, test_scenario)
        result = installer.mark_as_installed(installables)
        if not result.success:
            logging.warning("Failed to mark workloads components as installed for dry-run.")

    logging.info(test_scenario.pretty_print())

    runner = Runner(args.mode, system, test_scenario)
    register_signal_handlers(runner.cancel_on_signal)
    logging.info(f"Scenario results will be stored at: {runner.runner.scenario_root}")

    successful = False
    try:
        runner.runner.experiment_output.write()
        has_dse = any(tr.is_dse_job for tr in test_scenario.test_runs)
        if args.single_sbatch or not has_dse:  # in this mode cases are unrolled using grid search
            successful = handle_non_dse_job(runner, args)
            return 0

        if all(tr.is_dse_job for tr in test_scenario.test_runs):
            result = handle_dse_job(runner, args)
            successful = result == 0
            return result

        logging.error("Mixing DSE and non-DSE jobs is not allowed.")
        return 1
    finally:
        runner.runner.finish_output(successful)


def handle_generate_report(args: argparse.Namespace) -> int:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config, args.hook_dir or HOOK_ROOT)
    system, _, test_scenario = parser.parse(args.tests_dir, args.test_scenario)
    assert test_scenario is not None

    generate_reports(system, test_scenario, args.result_dir)

    logging.debug("Report generation completed.")

    return 0


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
