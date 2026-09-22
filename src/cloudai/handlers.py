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

import copy
import datetime
import json
import logging
import subprocess
import sys
import tempfile
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import toml

import cloudai.models.output
import cloudai.output
from cloudai.configurator.env_params import validate_domain_randomization_active
from cloudai.core import (
    BaseInstaller,
    CloudAIGymEnv,
    ConfigPaths,
    Installable,
    MissingTestError,
    Parser,
    Registry,
    Runner,
    System,
    SystemConfigParsingError,
    TestConfigParsingError,
    TestScenario,
    TestScenarioParsingError,
)
from cloudai.models.scenario import ReportConfig
from cloudai.models.workload import TestDefinition
from cloudai.parser import HOOK_ROOT
from cloudai.systems.slurm import SingleSbatchRunner, SlurmSystem
from cloudai.util import prepare_output_dir


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


def handle_dse_job(runner: Runner, mode: str) -> int:
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

    if mode == "run":
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


def handle_non_dse_job(runner: Runner) -> bool:
    successful = runner.run()
    generate_reports(runner.runner.system, runner.runner.test_scenario, runner.runner.scenario_root)
    logging.info("All jobs are complete.")
    return successful


def load_scenario(
    scenario: Path, system: Path, *, tests_dir: Path | None = None, hook_dir: Path | None = None
) -> tuple[System, list[TestDefinition], TestScenario]:
    """Parse an experiment without installation, scheduler queries, or process exits."""
    try:
        system.stat()
    except OSError as exc:
        raise SystemConfigParsingError(str(exc)) from exc
    parser = Parser(system, hook_dir or HOOK_ROOT, exit_on_error=False)
    parsed_system, tests, parsed_scenario = parser.parse(tests_dir, scenario)
    if parsed_scenario is None:
        raise TestScenarioParsingError("A test scenario is required.")
    return parsed_system, tests, parsed_scenario


def validate_experiment(system: System, scenario: TestScenario, single_sbatch: bool = False) -> None:
    """Check execution constraints shared by validation and execution."""
    validate_domain_randomization_active(scenario)
    if single_sbatch:
        if not isinstance(system, SlurmSystem):
            raise TestScenarioParsingError("Single sbatch is only supported for Slurm systems.")
        return
    dse_runs = [tr for tr in scenario.test_runs if tr.is_dse_job]
    if dse_runs and len(dse_runs) != len(scenario.test_runs):
        raise TestScenarioParsingError("Mixing DSE and non-DSE jobs is not allowed.")
    for tr in dse_runs:
        if tr.dependencies:
            raise TestScenarioParsingError("Dependencies are not supported for DSE jobs.")
        registry = Registry()
        if not registry.has_agent(tr.test.agent):
            raise TestScenarioParsingError(f"No agent available for type: {tr.test.agent}.")
        registry.get_agent(tr.test.agent).get_config_class()(**(tr.test.agent_config or {}))


def create_experiment_runner(
    system: System,
    scenario: TestScenario,
    *,
    dry_run: bool = False,
    single_sbatch: bool = False,
    result_dir: Path | None = None,
) -> Runner:
    """Create a runner without changing the registered defaults."""
    validate_experiment(system, scenario, single_sbatch)
    if prepare_output_dir(system.output_path) is None:
        raise OSError(f"Cannot prepare output directory: {system.output_path}")
    if dry_run:
        system.monitor_interval = 1
    return Runner(
        "dry-run" if dry_run else "run",
        system,
        scenario,
        runner_class=SingleSbatchRunner if single_sbatch else None,
        output_path=result_dir,
    )


def execute_experiment(runner: Runner, tests: list[TestDefinition], enable_cache_without_check: bool = False) -> int:
    """Install prerequisites, execute the scenario, and finalize its output."""
    system = runner.runner.system
    scenario = runner.runner.test_scenario
    successful = False
    try:
        runner.runner.experiment_output.write()
        system.update()
        logging.info("System Name: %s", system.name)
        logging.info("Scheduler: %s", system.scheduler)
        logging.info("Test Scenario Name: %s", scenario.name)
        installables, installer = prepare_installation(system, tests, scenario)
        if enable_cache_without_check or runner.runner.mode == "dry-run":
            result = installer.mark_as_installed(installables)
        else:
            result = installer.is_installed(installables)
        if runner.runner.mode == "run" and not result.success:
            logging.info("Not all workload components are installed. Installing...")
            result = installer.install(installables)
            if not result.success:
                raise RuntimeError(f"Failed to install workload components: {result.message}")
        elif runner.runner.mode == "dry-run" and not result.success:
            logging.warning("Failed to mark workload components as installed for dry-run.")
        logging.info(scenario.pretty_print())
        if isinstance(runner.runner, SingleSbatchRunner) or not any(tr.is_dse_job for tr in scenario.test_runs):
            successful = handle_non_dse_job(runner)
            return 0
        result_code = handle_dse_job(runner, runner.runner.mode)
        successful = result_code == 0
        return result_code
    finally:
        runner.runner.finish_output(successful)


@contextmanager
def _configuration_files(system: str | Path, scenario: str | Path | None = None):
    with tempfile.TemporaryDirectory(prefix="cloudai-config-") as temporary:
        system_path = system.expanduser().resolve() if isinstance(system, Path) else Path(temporary) / "system.toml"
        if isinstance(system, str):
            system_path.write_text(system, encoding="utf-8")
        scenario_path = Path(temporary) / "scenario.toml"
        if isinstance(scenario, Path):
            scenario_path = scenario.expanduser().resolve()
        elif isinstance(scenario, str):
            try:
                data = toml.loads(scenario)
            except toml.TomlDecodeError as exc:
                raise TestScenarioParsingError(str(exc)) from exc
            tests = data.get("Tests", [])
            if isinstance(tests, list):
                for test in tests:
                    path = test.get("path") if isinstance(test, dict) else None
                    if isinstance(path, str) and not Path(path).is_absolute():
                        raise TestScenarioParsingError(
                            "Relative test paths require a scenario file; pass a Path or use absolute test paths."
                        )
            scenario_path.write_text(scenario, encoding="utf-8")
        yield system_path, scenario_path


def run_experiment(
    scenario: str | Path,
    system: str | Path,
    wait: bool = True,
    *,
    tests_dir: Path | None = None,
    hook_dir: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    single_sbatch: bool = False,
    enable_cache_without_check: bool = False,
) -> cloudai.models.output.Experiment:
    """Run an experiment synchronously or hand it to an independent worker."""
    with _configuration_files(system, scenario) as (system_path, scenario_path):
        parsed_system, tests, parsed_scenario = load_scenario(
            scenario_path, system_path, tests_dir=tests_dir, hook_dir=hook_dir
        )
        validate_experiment(parsed_system, parsed_scenario, single_sbatch)
        parsed_system.output_path = (output_dir or parsed_system.output_path).expanduser().resolve()
        if prepare_output_dir(parsed_system.output_path) is None:
            raise OSError(f"Cannot prepare output directory: {parsed_system.output_path}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = Path(
            tempfile.mkdtemp(prefix=f"{parsed_scenario.name}_{timestamp}_", dir=parsed_system.output_path)
        ).resolve()
        if isinstance(system, str):
            system_path = result_dir / "system.toml"
            system_path.write_text(system, encoding="utf-8")
        if isinstance(scenario, str):
            scenario_path = result_dir / "scenario.toml"
            scenario_path.write_text(scenario, encoding="utf-8")
        parsed_scenario.config_paths = ConfigPaths(
            system_path=system_path,
            test_scenario_path=scenario_path,
            tests_dir_path=tests_dir.resolve() if tests_dir is not None else None,
        )
        runner = create_experiment_runner(
            parsed_system, parsed_scenario, dry_run=dry_run, single_sbatch=single_sbatch, result_dir=result_dir
        )
        output = runner.runner.experiment_output
        if wait:
            execute_experiment(runner, tests, enable_cache_without_check)
            return output.snapshot()
        output.experiment.status = "pending"
        output.experiment.start = None
        output.write()
        initial = get_experiment(result_dir)
        request = {
            "system": str(system_path),
            "scenario": str(scenario_path),
            "tests_dir": str(tests_dir.resolve()) if tests_dir is not None else None,
            "hook_dir": str((hook_dir or HOOK_ROOT).resolve()),
            "output_dir": str(parsed_system.output_path.resolve()),
            "dry_run": dry_run,
            "single_sbatch": single_sbatch,
            "enable_cache_without_check": enable_cache_without_check,
        }
        request_path = result_dir / "request.json"
        try:
            request_path.write_text(json.dumps(request), encoding="utf-8")
            command = [
                sys.executable,
                "-c",
                "import logging, pathlib, sys; import cloudai.handlers; "
                "logging.basicConfig(level=logging.INFO); "
                "cloudai.handlers.run_background(pathlib.Path(sys.argv[1]))",
                str(request_path),
            ]
            with (result_dir / "controller.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            threading.Thread(target=process.wait, daemon=True).start()
        except Exception:
            runner.runner.finish_output(False)
            raise
        return initial


def run_background(request_path: Path) -> None:
    """Execute a persisted request in its worker process."""
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        system, tests, scenario = load_scenario(
            Path(request["scenario"]),
            Path(request["system"]),
            tests_dir=Path(request["tests_dir"]) if request["tests_dir"] else None,
            hook_dir=Path(request["hook_dir"]),
        )
        system.output_path = Path(request["output_dir"])
        runner = create_experiment_runner(
            system,
            scenario,
            dry_run=request["dry_run"],
            single_sbatch=request["single_sbatch"],
            result_dir=request_path.parent,
        )
        execute_experiment(runner, tests, request["enable_cache_without_check"])
    except BaseException:
        logging.exception("Background experiment failed")
        experiment = get_experiment(request_path.parent)
        if experiment.status in ("pending", "running"):
            cloudai.output.ExperimentOutput(experiment, request_path.parent).finish(
                "failed", datetime.datetime.now(datetime.timezone.utc)
            )
        raise


def get_experiment(exp: str | Path) -> cloudai.models.output.Experiment:
    """Read a saved experiment from its result directory or JSON file."""
    path = Path(exp).expanduser()
    if path.is_dir():
        path /= "experiment.json"
    return cloudai.models.output.Experiment.model_validate_json(path.read_text(encoding="utf-8"))


def list_experiments(system: str | Path) -> list[tuple[str, Path]]:
    """Discover experiments beneath the configured results directory."""
    with _configuration_files(system) as (system_path, _):
        output_dir = Parser.parse_system(system_path).output_path.expanduser().resolve()
    if not output_dir.exists():
        return []
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    return [(get_experiment(path).id, path.parent) for path in sorted(output_dir.glob("*/experiment.json"))]


def validate_scenario(
    scenario: str | Path,
    system: str | Path,
    *,
    tests_dir: Path | None = None,
    hook_dir: Path | None = None,
    single_sbatch: bool = False,
) -> tuple[bool, dict[str, str]]:
    """Validate configuration and execution constraints without running workloads."""
    try:
        with _configuration_files(system, scenario) as (system_path, scenario_path):
            parsed_system, _, parsed_scenario = load_scenario(
                scenario_path, system_path, tests_dir=tests_dir, hook_dir=hook_dir
            )
            validate_experiment(parsed_system, parsed_scenario, single_sbatch)
    except SystemConfigParsingError as exc:
        return False, {"system": str(exc.__cause__ or exc)}
    except (TestConfigParsingError, TestScenarioParsingError, MissingTestError, ValueError, OSError) as exc:
        return False, {"scenario": str(exc)}
    return True, {}
