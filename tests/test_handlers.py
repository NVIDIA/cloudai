# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tarfile
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import Field

import cloudai.models.output
from cloudai.configurator import CloudAIGymEnv
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import (
    BaseAgent,
    BaseAgentConfig,
    GitRepo,
    InstallStatusResult,
    JobStatusResult,
    Parser,
    Registry,
    RewardOverrides,
    Runner,
    TestDependency,
    TestRun,
    TestScenario,
    TestScenarioParsingError,
)
from cloudai.handlers import (
    InstallationError,
    execute_experiment,
    handle_dse_job,
    handle_install_and_uninstall,
    prepare_installation,
    validate_domain_randomization_active,
    verify_system_configs,
    verify_test_configs,
    verify_test_scenarios,
)
from cloudai.models.scenario import ReportConfig
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.reporter import StatusReporter, TarballReporter
from cloudai.systems.slurm import SlurmRunner, SlurmSystem
from cloudai.systems.standalone import StandaloneRunner, StandaloneSystem
from cloudai.test_parser import TestParser


class StubAgentConfig(BaseAgentConfig):
    knob: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class StubAgent(BaseAgent):
    received_configs: ClassVar[list[StubAgentConfig]] = []
    supports_variable_environment: bool = True  # stands in for an env-aware learning agent

    def __init__(self, env, config: StubAgentConfig):
        self.env = env
        self.config = config
        self.max_steps = 0
        StubAgent.received_configs.append(config)

    @staticmethod
    def get_config_class() -> type[StubAgentConfig]:
        return StubAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]]:
        raise NotImplementedError

    def update_policy(self, _feedback: dict[str, Any]) -> None:
        return


@pytest.fixture
def stub_agent_name() -> Iterator[str]:
    registry = Registry()
    agent_name = "test_handlers_stub_agent"
    old_agent = registry.agents_map.get(agent_name)
    registry.update_agent(agent_name, StubAgent)
    StubAgent.received_configs.clear()
    yield agent_name
    StubAgent.received_configs.clear()
    if old_agent is None:
        del registry.agents_map[agent_name]
    else:
        registry.update_agent(agent_name, old_agent)


def test_runner_warns_about_legacy_create_runner_signature(slurm_system: SlurmSystem, dse_tr: TestRun) -> None:
    scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    with pytest.warns(DeprecationWarning, match="will be removed in CloudAI 1.9.1"):
        Runner(mode="dry-run", system=slurm_system, test_scenario=scenario)


def test_runner_uses_explicit_class_without_registered_scheduler(slurm_system: SlurmSystem, dse_tr: TestRun) -> None:
    slurm_system.scheduler = "custom"
    scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])

    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=scenario, runner_class=SlurmRunner)

    assert isinstance(runner.runner, SlurmRunner)


@pytest.mark.parametrize("dep", ["start_post_comp", "start_post_init", "end_post_comp"])
def test_dse_run_does_not_support_dependencies(
    slurm_system: SlurmSystem, dse_tr: TestRun, dep: str, caplog: pytest.LogCaptureFixture
) -> None:
    """
    DSE runs do not support dependencies.

    DSE engine re-uses BaseRunner by manually controlling test_run to execute. BaseRunner doesn't keep track of all jobs
    and their statuses, this information is not available between cases in a scenario or even between steps of a single
    test run.

    While it might be useful in future, today we have to explicitly forbid such configurations and report actionable
    error to users.
    """
    dse_tr.dependencies = {dep: TestDependency(test_run=dse_tr)}
    test_scenario: TestScenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)
    assert handle_dse_job(runner) == 1
    assert "Dependencies are not supported for DSE jobs, all cases run consecutively." in caplog.text
    assert "Please remove dependencies and re-run." in caplog.text


@pytest.mark.parametrize(
    "agent_config,expected",
    [
        (
            {
                "random_seed": 123,
                "start_action": "first",
                "knob": 7,
                "payload": {"alpha": 1, "beta": "value"},
            },
            {
                "random_seed": 123,
                "start_action": "first",
                "knob": 7,
                "payload": {"alpha": 1, "beta": "value"},
            },
        ),
        (
            None,
            {
                "random_seed": 42,
                "start_action": "random",
                "knob": 0,
                "payload": {},
            },
        ),
    ],
    ids=["overrides-agent-config", "uses-default-agent-config"],
)
def test_dse_run_uses_agent_config(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    stub_agent_name: str,
    agent_config: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    dse_tr.test.agent = stub_agent_name
    dse_tr.test.agent_config = agent_config
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)

    assert handle_dse_job(runner) == 0
    assert len(StubAgent.received_configs) == 1

    recorded = StubAgent.received_configs[0]
    assert recorded.start_action == expected["start_action"]
    assert recorded.knob == expected["knob"]
    assert recorded.payload == expected["payload"]
    assert recorded.random_seed == expected["random_seed"]


@pytest.mark.parametrize("mode", ["install", "run"])
@pytest.mark.parametrize("already_installed", [True, False])
@pytest.mark.parametrize("install_success", [True, False])
def test_install_and_run_share_installation(
    mode: str,
    already_installed: bool,
    install_success: bool,
    standalone_system: StandaloneSystem,
    base_tr: TestRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = TestScenario(name="scenario", test_runs=[base_tr])
    installer = MagicMock()
    installer.is_installed.return_value = InstallStatusResult(already_installed)
    installer.install.return_value = InstallStatusResult(install_success, "installation outcome")
    prepare = MagicMock(return_value=([], installer))
    monkeypatch.setattr("cloudai.handlers.prepare_installation", prepare)
    failed = not already_installed and not install_success

    if mode == "install":
        parser = MagicMock()
        parser.parse.return_value = (standalone_system, [], scenario)
        monkeypatch.setattr("cloudai.handlers.Parser", lambda *_: parser)
        args = argparse.Namespace(mode=mode, system_config=None, hook_dir=None, tests_dir=None, test_scenario=None)
        assert handle_install_and_uninstall(args) == int(failed)
    else:
        runner = Runner(mode, standalone_system, scenario, runner_class=StandaloneRunner)
        monkeypatch.setattr(Runner, "run", lambda _: True)
        monkeypatch.setattr("cloudai.handlers.generate_reports", lambda *_: None)
        if failed:
            with pytest.raises(InstallationError, match="installation outcome"):
                execute_experiment(runner, [])
        else:
            assert execute_experiment(runner, [])

    prepare.assert_called_once_with(standalone_system, [], scenario)
    installer.is_installed.assert_called_once_with([])
    installer.mark_as_installed.assert_not_called()
    if already_installed:
        installer.install.assert_not_called()
    else:
        installer.install.assert_called_once_with([])


@pytest.mark.parametrize("mode", ["run", "dry-run"])
@pytest.mark.parametrize("cache_without_check", [True, False])
def test_execution_installation_cache_modes(
    mode: str,
    cache_without_check: bool,
    standalone_system: StandaloneSystem,
    base_tr: TestRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = TestScenario(name="scenario", test_runs=[base_tr])
    installer = MagicMock()
    installer.is_installed.return_value = InstallStatusResult(True)
    installer.mark_as_installed.return_value = InstallStatusResult(True)
    monkeypatch.setattr("cloudai.handlers.prepare_installation", lambda *_: ([], installer))
    monkeypatch.setattr(Runner, "run", lambda _: True)
    monkeypatch.setattr("cloudai.handlers.generate_reports", lambda *_: None)
    runner = Runner(mode, standalone_system, scenario, runner_class=StandaloneRunner)

    assert execute_experiment(runner, [], enable_cache_without_check=cache_without_check)

    if cache_without_check:
        installer.is_installed.assert_not_called()
    else:
        installer.is_installed.assert_called_once_with([])
    assert installer.mark_as_installed.call_count == int(cache_without_check) + int(mode == "dry-run")
    installer.install.assert_not_called()


def test_prepare_installation_includes_hook_installables(slurm_system: SlurmSystem) -> None:
    parent_repo = GitRepo(url="./parent", commit="main")
    pre_repo = GitRepo(url="./pre", commit="main")
    post_repo = GitRepo(url="./post", commit="main")
    parent_run = TestRun(
        name="parent",
        test=TestDefinition(
            name="parent",
            description="parent",
            test_template_name="template",
            cmd_args=CmdArgs(),
            git_repos=[parent_repo],
        ),
        num_nodes=1,
        nodes=[],
        pre_test=TestScenario(
            name="pre",
            test_runs=[
                TestRun(
                    name="pre",
                    test=TestDefinition(
                        name="pre",
                        description="pre",
                        test_template_name="template",
                        cmd_args=CmdArgs(),
                        git_repos=[pre_repo],
                    ),
                    num_nodes=1,
                    nodes=[],
                )
            ],
        ),
        post_test=TestScenario(
            name="post",
            test_runs=[
                TestRun(
                    name="post",
                    test=TestDefinition(
                        name="post",
                        description="post",
                        test_template_name="template",
                        cmd_args=CmdArgs(),
                        git_repos=[post_repo],
                    ),
                    num_nodes=1,
                    nodes=[],
                )
            ],
        ),
    )

    installables, _ = prepare_installation(slurm_system, [], TestScenario(name="scenario", test_runs=[parent_run]))

    assert parent_repo in installables
    assert pre_repo in installables
    assert post_repo in installables


def test_dse_run_cache(base_tr: TestRun, tmp_path, caplog: pytest.LogCaptureFixture):
    base_tr.test.cmd_args.candidate = [1, 1, 2]
    base_tr.test.agent = "grid_search"
    base_tr.test.agent_steps = 3

    inner_runner = MagicMock()
    inner_runner.system = MagicMock()
    inner_runner.scenario_root = tmp_path / "scenario"
    inner_runner.test_scenario = TestScenario(name="test_scenario", test_runs=[base_tr])
    inner_runner.jobs = {}
    inner_runner.testrun_to_job_map = {}

    def _job_output_path(tr: TestRun, create: bool = True):
        output_path = inner_runner.scenario_root / tr.name / f"{tr.current_iteration}" / f"{tr.step}"
        if create:
            output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    inner_runner.get_job_output_path.side_effect = _job_output_path

    runner = MagicMock()
    runner.runner = inner_runner

    trajectory_dir = inner_runner.scenario_root / base_tr.name / f"{base_tr.current_iteration}"

    # run test
    with caplog.at_level("INFO"):
        assert handle_dse_job(runner) == 0

    reporter = StatusReporter(
        inner_runner.system,
        TestScenario(name="test_scenario", test_runs=[base_tr]),
        inner_runner.scenario_root,
        ReportConfig(),
    )
    reporter.load_test_runs()

    assert inner_runner.run.call_count == 2
    assert (trajectory_dir / "1").exists()
    assert not (trajectory_dir / "2").exists()
    assert (trajectory_dir / "3").exists()
    assert caplog.text.count("Retrieved cached result from") == 1

    actual_trajectory = pd.read_csv(trajectory_dir / "trajectory.csv")
    expected_trajectory = pd.DataFrame(
        data=[
            [1, "{'candidate': 1}", -1.0, "[-1.0]"],
            [2, "{'candidate': 1}", -1.0, "[-1.0]"],
            [3, "{'candidate': 2}", -1.0, "[-1.0]"],
        ],
        columns=["step", "action", "reward", "observation"],
    )
    pd.testing.assert_frame_equal(actual_trajectory, expected_trajectory)

    assert [tr.step for tr in reporter.trs] == [1, 3]


def test_rewards_nested() -> None:
    cfg = BaseAgentConfig.model_validate({"rewards": {"constraint_failure": -2.5, "metric_failure": 0.0}})
    assert cfg.rewards == RewardOverrides(constraint_failure=-2.5, metric_failure=0.0)


def test_verify_test_configs_logs_failure_details(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broken_test = tmp_path / "broken_test.toml"
    broken_test.write_text('name = "first"\nname = "second"\n')

    with caplog.at_level("INFO"):
        nfailed = verify_test_configs([broken_test])

    assert nfailed == 1
    assert str(broken_test) in caplog.text
    assert "duplicate TOML key 'name'" in caplog.text
    assert "1 out of 1 test configurations have issues." in caplog.text


def test_verify_system_configs_logs_failure_details(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broken_system = tmp_path / "broken_system.toml"
    broken_system.write_text('scheduler = "slurm"\nscheduler = "kubernetes"\n')

    with caplog.at_level("INFO"):
        nfailed = verify_system_configs([broken_system])

    assert nfailed == 1
    assert str(broken_system) in caplog.text
    assert "duplicate TOML key 'scheduler'" in caplog.text
    assert "1 out of 1 system configurations have issues." in caplog.text


def test_verify_test_scenarios_logs_failure_details(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broken_scenario = tmp_path / "broken_scenario.toml"
    broken_scenario.write_text('name = "first"\nname = "second"\n[[Tests]]\nid = "t1"\ntest_name = "demo"\n')

    with caplog.at_level("INFO"):
        nfailed = verify_test_scenarios([broken_scenario], [], [], [])

    assert nfailed == 1
    assert str(broken_scenario) in caplog.text
    assert "duplicate TOML key 'name'" in caplog.text
    assert "1 out of 1 test scenarios have issues." in caplog.text


def test_verify_test_scenarios_does_not_blame_scenarios_for_an_unreferenced_unparseable_test(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_unregistered_agent_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    bad_test = tmp_path / "bad_agent_test.toml"
    write_unregistered_agent_test(bad_test, "bad_agent_test")
    scenarios = []
    for index in range(3):
        scenario = tmp_path / f"scenario_{index}.toml"
        write_scenario(scenario, "good_test")
        scenarios.append(scenario)

    assert verify_test_scenarios(scenarios, [good_test, bad_test], [], []) == 0


def test_verify_test_scenarios_parses_test_configs_once_regardless_of_scenario_count(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    driven: list[TestParser] = []
    parse_all = TestParser.parse_all

    def spying_parse_all(self: TestParser):
        driven.append(self)
        return parse_all(self)

    monkeypatch.setattr(TestParser, "parse_all", spying_parse_all)

    def parse_all_calls_for(scenario_count: int) -> int:
        scenarios = []
        for index in range(scenario_count):
            scenario = tmp_path / f"scenario_{scenario_count}_{index}.toml"
            write_scenario(scenario, "good_test")
            scenarios.append(scenario)
        driven.clear()
        assert verify_test_scenarios(scenarios, [good_test], [], []) == 0
        return len(driven)

    assert parse_all_calls_for(1) == parse_all_calls_for(5)


def test_verify_test_scenarios_reports_an_unparseable_hook_test_no_scenario_references(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_unregistered_agent_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    bad_hook_test = tmp_path / "bad_hook_test.toml"
    write_unregistered_agent_test(bad_hook_test, "bad_hook_test")
    scenario = tmp_path / "scenario.toml"
    write_scenario(scenario, "good_test")

    with caplog.at_level("INFO"):
        nfailed = verify_test_scenarios([scenario], [good_test], [], [bad_hook_test])

    assert nfailed > 0
    assert str(bad_hook_test) in caplog.text
    assert "is not registered" in caplog.text


def test_verify_test_scenarios_blames_the_scenario_that_references_an_unparseable_test(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_unregistered_agent_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    bad_test = tmp_path / "bad_agent_test.toml"
    write_unregistered_agent_test(bad_test, "bad_agent_test")
    scenario = tmp_path / "uses_bad.toml"
    write_scenario(scenario, "bad_agent_test")

    with caplog.at_level("INFO"):
        nfailed = verify_test_scenarios([scenario], [good_test, bad_test], [], [])

    assert nfailed == 1
    messages = [record.getMessage() for record in caplog.records]
    blame = [msg for msg in messages if msg.startswith(f"Failed to verify Test Scenario: {scenario}:")]
    assert len(blame) == 1
    assert str(bad_test) in blame[0]
    assert "is not registered" in blame[0]
    assert "is not defined" not in blame[0]


def test_verify_test_scenarios_blames_the_scenario_that_names_a_test_that_exists_nowhere(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    scenario = tmp_path / "uses_missing.toml"
    write_scenario(scenario, "no_such_test_anywhere")

    with caplog.at_level("INFO"):
        nfailed = verify_test_scenarios([scenario], [good_test], [], [])

    assert nfailed == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(str(scenario) in msg and "no_such_test_anywhere" in msg for msg in messages)


def test_verify_test_scenarios_reports_a_duplicate_test_name_to_every_scenario_without_raising(
    tmp_path: Path,
    write_parseable_test: Callable[[Path, str], None],
    write_scenario: Callable[[Path, str], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_test = tmp_path / "good_test.toml"
    write_parseable_test(good_test, "good_test")
    duplicate = tmp_path / "good_test_copy.toml"
    write_parseable_test(duplicate, "good_test")
    scenarios = []
    for index in range(2):
        scenario = tmp_path / f"scenario_{index}.toml"
        write_scenario(scenario, "good_test")
        scenarios.append(scenario)

    with caplog.at_level("INFO"):
        nfailed = verify_test_scenarios(scenarios, [good_test, duplicate], [], [])

    assert nfailed == 2
    messages = [record.getMessage() for record in caplog.records]
    blamed = [msg for msg in messages if str(good_test) in msg and str(duplicate) in msg]
    assert len(blamed) == 2


class CustomRunStubAgentConfig(BaseAgentConfig):
    pass


class CustomRunStubAgent(BaseAgent):
    """Stub agent that overrides ``run()`` to drive its own training (e.g. RLlib-like)."""

    run_calls: ClassVar[int] = 0
    run_returns: ClassVar[int] = 0
    run_raises: ClassVar[Optional[BaseException]] = None

    def __init__(self, env, config: CustomRunStubAgentConfig):
        self.env = env
        self.config = config
        self.max_steps = 0

    @staticmethod
    def get_config_class() -> type[CustomRunStubAgentConfig]:
        return CustomRunStubAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]]:
        raise AssertionError("select_action must not be called when run() is overridden")

    def update_policy(self, _feedback: dict[str, Any]) -> None:
        return

    def run(self) -> int:
        CustomRunStubAgent.run_calls += 1
        if CustomRunStubAgent.run_raises is not None:
            raise CustomRunStubAgent.run_raises
        return CustomRunStubAgent.run_returns


@pytest.fixture
def custom_run_agent_name() -> Iterator[str]:
    registry = Registry()
    agent_name = "test_handlers_custom_run_agent"
    old_agent = registry.agents_map.get(agent_name)
    registry.update_agent(agent_name, CustomRunStubAgent)
    CustomRunStubAgent.run_calls = 0
    CustomRunStubAgent.run_returns = 0
    CustomRunStubAgent.run_raises = None
    yield agent_name
    CustomRunStubAgent.run_calls = 0
    CustomRunStubAgent.run_returns = 0
    CustomRunStubAgent.run_raises = None
    if old_agent is None:
        del registry.agents_map[agent_name]
    else:
        registry.update_agent(agent_name, old_agent)


def test_handle_dse_job_invokes_agent_run(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``handle_dse_job`` must delegate orchestration to ``agent.run()`` (polymorphism)."""
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)
    update_output = MagicMock()
    monkeypatch.setattr(CloudAIGymEnv, "update_output", update_output)

    assert handle_dse_job(runner) == 0
    assert CustomRunStubAgent.run_calls == 1
    assert update_output.call_count == 2


def test_handle_dse_job_propagates_agent_run_nonzero_rc(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
) -> None:
    """A non-zero rc from ``agent.run()`` must flow through to the caller via ``err |= rc``."""
    CustomRunStubAgent.run_returns = 1
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)

    assert handle_dse_job(runner) == 1
    assert CustomRunStubAgent.run_calls == 1


def test_handle_dse_job_accumulates_nonzero_rc_and_continues(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
) -> None:
    """Graceful failure: a non-zero rc accumulates via ``err |= rc`` and the sweep continues.

    A ``run()`` that returns a non-zero rc (the convention for recoverable failures, e.g.
    ``rllib_run`` catching a training error) must not abort the scenario: the remaining
    independent ``TestRun`` still executes and the accumulated error is reported.
    """
    CustomRunStubAgent.run_returns = 1
    dse_tr.test.agent = custom_run_agent_name
    second_tr = copy.deepcopy(dse_tr)
    second_tr.name = "dse_second"
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr, second_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)

    assert handle_dse_job(runner) == 1
    assert CustomRunStubAgent.run_calls == 2


def test_handle_dse_job_propagates_agent_run_exception(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard failure: an exception out of ``agent.run()`` propagates instead of being swallowed.

    Unexpected exceptions signal framework/agent bugs and must surface (hard-fail) rather than
    be masked as a non-zero rc; recoverable failures are expected to return a non-zero rc.
    """
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)
    update_output = MagicMock()
    monkeypatch.setattr(CloudAIGymEnv, "update_output", update_output)

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner)
    assert CustomRunStubAgent.run_calls == 1
    assert update_output.call_count == 2


def test_handle_dse_job_hard_fail_aborts_remaining_runs(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
) -> None:
    """A raising ``agent.run()`` aborts the scenario; subsequent ``TestRun`` are not started."""
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    second_tr = copy.deepcopy(dse_tr)
    second_tr.name = "dse_second"
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr, second_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner)
    assert CustomRunStubAgent.run_calls == 1


def test_handle_dse_job_documents_failure_before_raising(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
    tmp_path: Path,
) -> None:
    """On a hard-fail, the aborting error is documented, then re-raised."""
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)
    runner.runner.scenario_root = tmp_path

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner)

    failure_report = tmp_path / "dse_failure.txt"
    assert failure_report.exists()
    contents = failure_report.read_text()
    assert "RuntimeError" in contents
    assert "agent blew up" in contents


@pytest.mark.parametrize("mode", ["run", "dry-run"])
@pytest.mark.parametrize("successful", [True, False])
def test_standalone_archive_contains_final_experiment(
    mode: str,
    successful: bool,
    standalone_system: StandaloneSystem,
    base_tr: TestRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = TestScenario(name="test_scenario", test_runs=[base_tr])
    runner = Runner(mode, standalone_system, scenario, runner_class=StandaloneRunner)
    monkeypatch.setattr(
        "cloudai.handlers.prepare_installation",
        lambda *_: ([], MagicMock(is_installed=lambda _: InstallStatusResult(True))),
    )
    monkeypatch.setattr(Registry, "ordered_scenario_reports", lambda _: [("tarball", TarballReporter)])

    monkeypatch.setattr(TestDefinition, "was_run_successful", lambda *_: JobStatusResult(successful))

    def run(runner: Runner) -> bool:
        (runner.runner.scenario_root / base_tr.name / "0").mkdir(parents=True)
        return successful

    monkeypatch.setattr(Runner, "run", run)
    assert execute_experiment(runner, []) is successful

    results_root = runner.runner.scenario_root
    local = cloudai.models.output.Experiment.model_validate_json((results_root / "experiment.json").read_text())
    assert local.status == ("completed" if successful else "failed")
    assert local.finish is not None
    if successful:
        assert not Path(f"{results_root}.tgz").exists()
        return
    with tarfile.open(f"{results_root}.tgz", "r:gz") as tar:
        archived_file = tar.extractfile(f"{results_root.name}/experiment.json")
        assert archived_file is not None
        archived = cloudai.models.output.Experiment.model_validate_json(archived_file.read())

    assert archived == local
    assert archived.status == ("completed" if successful else "failed")
    assert archived.finish is not None


def test_dse_failure_report_contains_final_experiment(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a hard-fail, final output and the aborting error are reported before re-raising."""
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="run", system=slurm_system, test_scenario=test_scenario, runner_class=SlurmRunner)
    results_root = runner.runner.scenario_root
    (results_root / dse_tr.name / "0" / "0").mkdir(parents=True)
    monkeypatch.setattr(
        "cloudai.handlers.prepare_installation",
        lambda *_: ([], MagicMock(is_installed=lambda _: InstallStatusResult(True))),
    )
    monkeypatch.setattr(Registry, "ordered_scenario_reports", lambda _: [("tarball", TarballReporter)])

    with pytest.raises(RuntimeError, match="agent blew up"):
        execute_experiment(runner, [])

    failure_report = results_root / "dse_failure.txt"
    assert failure_report.exists()
    contents = failure_report.read_text()
    assert "RuntimeError" in contents
    assert "agent blew up" in contents
    with tarfile.open(f"{results_root}.tgz", "r:gz") as tar:
        archived_file = tar.extractfile(f"{results_root.name}/experiment.json")
        assert archived_file is not None
        archived = cloudai.models.output.Experiment.model_validate_json(archived_file.read())
    local = cloudai.models.output.Experiment.model_validate_json((results_root / "experiment.json").read_text())
    assert archived == local
    assert archived.status == "failed"
    assert archived.finish is not None
    assert [tr.name for tr in runner.runner.test_scenario.test_runs] == [dse_tr.name]


def test_dse_dry_run_generates_reports(
    slurm_system: SlurmSystem, dse_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner("dry-run", slurm_system, scenario, runner_class=SlurmRunner)
    monkeypatch.setattr(
        "cloudai.handlers.prepare_installation",
        lambda *_: ([], MagicMock(mark_as_installed=lambda _: InstallStatusResult(True))),
    )
    slurm_system.reports = {
        "per_test": ReportConfig(enable=False),
        "status": ReportConfig(enable=True),
        "dse": ReportConfig(enable=True),
    }

    assert execute_experiment(runner, [], enable_cache_without_check=True)

    root = runner.runner.scenario_root
    assert (root / "test_scenario.html").is_file()
    assert (root / dse_tr.name / "0" / "trajectory.csv").is_file()
    assert not (root / "test_scenario-dse-report.html").exists()
    assert not (root / dse_tr.name / "0" / f"{dse_tr.name}.toml").exists()
    assert [tr.name for tr in scenario.test_runs] == [dse_tr.name]


def test_validate_domain_randomization_active_rejects_non_dse(base_tr: TestRun) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    scenario = TestScenario(name="s", test_runs=[base_tr])
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_domain_randomization_active(scenario)


def test_validate_domain_randomization_active_rejects_grid_search(dse_tr: TestRun) -> None:
    """A DSE job on grid_search exhaustively searches the space, so env_params are noise -> reject."""
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    assert dse_tr.is_dse_job is True  # it IS a DSE job...
    assert dse_tr.test.agent == "grid_search"  # ...but grid_search does not sample env_params
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_domain_randomization_active(TestScenario(name="s", test_runs=[dse_tr]))


def test_validate_domain_randomization_active_rejects_non_sampling_agent(
    dse_tr: TestRun, stub_agent_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check keys on the agent capability, not the name: a non-grid agent that opts out is rejected too."""
    monkeypatch.setattr(StubAgent, "supports_variable_environment", False)
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name
    assert dse_tr.is_dse_job is True and dse_tr.test.agent != "grid_search"
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_domain_randomization_active(TestScenario(name="s", test_runs=[dse_tr]))


def test_validate_domain_randomization_active_defers_unknown_agent(dse_tr: TestRun) -> None:
    """An unknown agent is not flagged here; it is deferred to the dedicated agent-resolution error."""
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = "does_not_exist_agent"
    assert dse_tr.is_dse_job is True
    assert dse_tr.test.agent not in Registry().agents_map  # precondition: agent is truly unknown
    validate_domain_randomization_active(TestScenario(name="s", test_runs=[dse_tr]))  # no exception == deferred


def test_validate_domain_randomization_active_allows_dse_run(dse_tr: TestRun, stub_agent_name: str) -> None:
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name  # an env-aware agent (supports_variable_environment=True) consumes env_params
    assert dse_tr.is_dse_job is True  # precondition: DSE + env-aware agent + env_params is allowed
    validate_domain_randomization_active(TestScenario(name="s", test_runs=[dse_tr]))  # no exception == pass


def test_validate_domain_randomization_active_allows_num_nodes_sweep(base_tr: TestRun, stub_agent_name: str) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    base_tr.test.agent = stub_agent_name
    base_tr.num_nodes = [1, 2]
    assert base_tr.is_dse_job is True  # a num_nodes list sweep makes it DSE, so env_params is allowed
    validate_domain_randomization_active(TestScenario(name="s", test_runs=[base_tr]))  # no exception == pass


def test_validate_domain_randomization_active_allows_non_dse_without_env_params(base_tr: TestRun) -> None:
    assert base_tr.is_dse_job is False  # precondition: not DSE, but also no env_params declared
    assert not base_tr.test.env_params
    validate_domain_randomization_active(TestScenario(name="s", test_runs=[base_tr]))  # no exception == pass


def test_verify_test_scenarios_rejects_env_params_without_dse(
    base_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    bad = TestScenario(name="s", test_runs=[base_tr])
    monkeypatch.setattr(Parser, "parse_test_scenario", lambda *a, **k: bad)
    assert verify_test_scenarios([Path("dummy.toml")], [], [], []) == 1


def test_verify_test_scenarios_allows_env_params_with_dse(
    dse_tr: TestRun, stub_agent_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name  # learning agent (not grid_search)
    good = TestScenario(name="s", test_runs=[dse_tr])
    monkeypatch.setattr(Parser, "parse_test_scenario", lambda *a, **k: good)
    assert verify_test_scenarios([Path("dummy.toml")], [], [], []) == 0
