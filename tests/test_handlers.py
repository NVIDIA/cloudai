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
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import Field

from cloudai.cli.handlers import (
    _dispatch_agent_driven_run,
    handle_dse_job,
    validate_dse_env_params,
    verify_system_configs,
    verify_test_configs,
    verify_test_scenarios,
)
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import (
    BaseAgent,
    BaseAgentConfig,
    CmdArgs,
    Parser,
    Registry,
    RewardOverrides,
    Runner,
    TestDefinition,
    TestDependency,
    TestRun,
    TestScenario,
    TestScenarioParsingError,
)
from cloudai.models.scenario import ReportConfig
from cloudai.reporter import StatusReporter
from cloudai.systems.slurm.slurm_system import SlurmSystem


class StubAgentConfig(BaseAgentConfig):
    knob: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class StubAgent(BaseAgent):
    received_configs: ClassVar[list[StubAgentConfig]] = []
    samples_env_params: bool = True  # stands in for an env-aware learning agent

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
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)
    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 1
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
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 0
    assert len(StubAgent.received_configs) == 1

    recorded = StubAgent.received_configs[0]
    assert recorded.start_action == expected["start_action"]
    assert recorded.knob == expected["knob"]
    assert recorded.payload == expected["payload"]
    assert recorded.random_seed == expected["random_seed"]


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
        assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 0

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
) -> None:
    """``handle_dse_job`` must delegate orchestration to ``agent.run()`` (polymorphism)."""
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 0
    assert CustomRunStubAgent.run_calls == 1


def test_handle_dse_job_propagates_agent_run_nonzero_rc(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
) -> None:
    """A non-zero rc from ``agent.run()`` must flow through to the caller via ``err |= rc``."""
    CustomRunStubAgent.run_returns = 1
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 1
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
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 1
    assert CustomRunStubAgent.run_calls == 2


def test_handle_dse_job_propagates_agent_run_exception(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
) -> None:
    """Hard failure: an exception out of ``agent.run()`` propagates instead of being swallowed.

    Unexpected exceptions signal framework/agent bugs and must surface (hard-fail) rather than
    be masked as a non-zero rc; recoverable failures are expected to return a non-zero rc.
    """
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner, argparse.Namespace(mode="dry-run"))
    assert CustomRunStubAgent.run_calls == 1


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
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner, argparse.Namespace(mode="dry-run"))
    assert CustomRunStubAgent.run_calls == 1


def test_handle_dse_job_documents_failure_in_reports_before_raising(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    custom_run_agent_name: str,
    tmp_path: Path,
) -> None:
    """On a hard-fail, reports are still generated and the aborting error is documented, then re-raised."""
    CustomRunStubAgent.run_raises = RuntimeError("agent blew up")
    dse_tr.test.agent = custom_run_agent_name
    test_scenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="run", system=slurm_system, test_scenario=test_scenario)
    runner.runner.scenario_root = tmp_path

    with pytest.raises(RuntimeError, match="agent blew up"):
        handle_dse_job(runner, argparse.Namespace(mode="run"))

    failure_report = tmp_path / "dse_failure.txt"
    assert failure_report.exists()
    contents = failure_report.read_text()
    assert "RuntimeError" in contents
    assert "agent blew up" in contents


def test_validate_dse_env_params_rejects_non_dse(base_tr: TestRun) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    scenario = TestScenario(name="s", test_runs=[base_tr])
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_dse_env_params(scenario)


def test_validate_dse_env_params_rejects_grid_search(dse_tr: TestRun) -> None:
    """A DSE job on grid_search exhaustively searches the space, so env_params are noise -> reject."""
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    assert dse_tr.is_dse_job is True  # it IS a DSE job...
    assert dse_tr.test.agent == "grid_search"  # ...but grid_search does not sample env_params
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_dse_env_params(TestScenario(name="s", test_runs=[dse_tr]))


def test_validate_dse_env_params_rejects_non_sampling_agent(
    dse_tr: TestRun, stub_agent_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check keys on the agent capability, not the name: a non-grid agent that opts out is rejected too."""
    monkeypatch.setattr(StubAgent, "samples_env_params", False)
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name
    assert dse_tr.is_dse_job is True and dse_tr.test.agent != "grid_search"
    with pytest.raises(TestScenarioParsingError, match="no agent will sample them"):
        validate_dse_env_params(TestScenario(name="s", test_runs=[dse_tr]))


def test_validate_dse_env_params_defers_unknown_agent(dse_tr: TestRun) -> None:
    """An unknown agent is not flagged here; it is deferred to the dedicated agent-resolution error."""
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = "does_not_exist_agent"
    assert dse_tr.is_dse_job is True
    assert dse_tr.test.agent not in Registry().agents_map  # precondition: agent is truly unknown
    validate_dse_env_params(TestScenario(name="s", test_runs=[dse_tr]))  # no exception == deferred


def test_validate_dse_env_params_allows_dse_run(dse_tr: TestRun, stub_agent_name: str) -> None:
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name  # an env-aware agent (samples_env_params=True) consumes env_params
    assert dse_tr.is_dse_job is True  # precondition: DSE + env-aware agent + env_params is allowed
    validate_dse_env_params(TestScenario(name="s", test_runs=[dse_tr]))  # no exception == pass


def test_validate_dse_env_params_allows_num_nodes_sweep(base_tr: TestRun, stub_agent_name: str) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    base_tr.test.agent = stub_agent_name
    base_tr.num_nodes = [1, 2]
    assert base_tr.is_dse_job is True  # a num_nodes list sweep makes it DSE, so env_params is allowed
    validate_dse_env_params(TestScenario(name="s", test_runs=[base_tr]))  # no exception == pass


def test_validate_dse_env_params_allows_non_dse_without_env_params(base_tr: TestRun) -> None:
    assert base_tr.is_dse_job is False  # precondition: not DSE, but also no env_params declared
    assert not base_tr.test.env_params
    validate_dse_env_params(TestScenario(name="s", test_runs=[base_tr]))  # no exception == pass


def test_verify_test_scenarios_rejects_env_params_without_dse(
    base_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    bad = TestScenario(name="s", test_runs=[base_tr])
    monkeypatch.setattr(Parser, "parse_tests", lambda *a, **k: [])
    monkeypatch.setattr(Parser, "parse_hooks", lambda *a, **k: {})
    monkeypatch.setattr(Parser, "parse_test_scenario", lambda *a, **k: bad)
    assert verify_test_scenarios([Path("dummy.toml")], [], [], []) == 1


def test_verify_test_scenarios_allows_env_params_with_dse(
    dse_tr: TestRun, stub_agent_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dse_tr.test.env_params = {"ball_speed": EnvParamSpec()}
    dse_tr.test.agent = stub_agent_name  # learning agent (not grid_search)
    good = TestScenario(name="s", test_runs=[dse_tr])
    monkeypatch.setattr(Parser, "parse_tests", lambda *a, **k: [])
    monkeypatch.setattr(Parser, "parse_hooks", lambda *a, **k: {})
    monkeypatch.setattr(Parser, "parse_test_scenario", lambda *a, **k: good)
    assert verify_test_scenarios([Path("dummy.toml")], [], [], []) == 0


def test_validate_dse_env_params_allows_live_rl_run() -> None:
    cmd_args = CmdArgs.model_validate({"live_rl_mode": True})
    tdef = TestDefinition(name="lr", description="d", test_template_name="tt", cmd_args=cmd_args)
    tdef.env_params = {"ball_speed": EnvParamSpec()}
    tr = TestRun(name="lr", test=tdef, num_nodes=1, nodes=[])
    assert tr.is_dse_job is False  # precondition: live-RL is not a DSE sweep, but is agent-driven
    validate_dse_env_params(TestScenario(name="s", test_runs=[tr]))  # no exception == pass


def _routing_tr(name: str, *, live_rl: bool = False, num_nodes: Any = 1) -> TestRun:
    cmd_args = CmdArgs.model_validate({"live_rl_mode": True}) if live_rl else CmdArgs()
    tdef = TestDefinition(name=name, description="d", test_template_name="tt", cmd_args=cmd_args)
    return TestRun(name=name, test=tdef, num_nodes=num_nodes, nodes=[])


@pytest.mark.parametrize(
    "live_rl, num_nodes, expected",
    [
        pytest.param(False, 1, False, id="plain-job"),
        pytest.param(True, 1, True, id="live-rl-job"),
        pytest.param(False, [1, 2], True, id="dse-via-num-nodes"),
    ],
)
def test_is_agent_driven(live_rl: bool, num_nodes: Any, expected: bool) -> None:
    assert _routing_tr("tr", live_rl=live_rl, num_nodes=num_nodes).is_agent_driven is expected


def _run_routing(test_runs: list[TestRun], *, single_sbatch: bool = False) -> tuple[int, MagicMock, MagicMock]:
    scenario = TestScenario(name="s", test_runs=test_runs)
    runner = MagicMock()
    with (
        patch("cloudai.cli.handlers.handle_dse_job", return_value=0) as dse,
        patch("cloudai.cli.handlers.handle_non_dse_job") as non_dse,
    ):
        rc = _dispatch_agent_driven_run(argparse.Namespace(single_sbatch=single_sbatch), runner, scenario)
    return rc, dse, non_dse


def test_handle_dry_run_routes_live_rl_to_dse_handler() -> None:
    """A live_rl_mode run (no TOML sweep, so not is_dse_job) must reach handle_dse_job (agent.run())."""
    rc, dse, non_dse = _run_routing([_routing_tr("live", live_rl=True)])
    dse.assert_called_once()
    non_dse.assert_not_called()
    assert rc == 0


def test_handle_dry_run_routes_plain_job_to_non_dse_handler() -> None:
    rc, dse, non_dse = _run_routing([_routing_tr("plain")])
    non_dse.assert_called_once()
    dse.assert_not_called()
    assert rc == 0


def test_handle_dry_run_mixed_live_rl_and_plain_errors() -> None:
    rc, dse, non_dse = _run_routing([_routing_tr("live", live_rl=True), _routing_tr("plain")])
    assert rc == 1
    dse.assert_not_called()
    non_dse.assert_not_called()


def test_handle_dry_run_single_sbatch_with_live_rl_errors() -> None:
    """single_sbatch has no live-RL path; the combination must hard-error, not silently run static.

    Tracked for a proper routing rework: https://github.com/NVIDIA/cloudai/issues/937.
    """
    rc, dse, non_dse = _run_routing([_routing_tr("live", live_rl=True)], single_sbatch=True)
    assert rc == 1
    dse.assert_not_called()
    non_dse.assert_not_called()


def test_handle_dry_run_single_sbatch_with_plain_job_grid_unrolls() -> None:
    rc, dse, non_dse = _run_routing([_routing_tr("plain")], single_sbatch=True)
    assert rc == 0
    non_dse.assert_called_once()
    dse.assert_not_called()
