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
from pathlib import Path
from typing import Any, ClassVar, Iterator
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import Field

from cloudai.cli.handlers import handle_dse_job, verify_system_configs, verify_test_configs, verify_test_scenarios
from cloudai.core import (
    BaseAgent,
    BaseAgentConfig,
    Registry,
    RewardOverrides,
    Runner,
    TestDependency,
    TestRun,
    TestScenario,
)
from cloudai.models.scenario import ReportConfig
from cloudai.reporter import StatusReporter
from cloudai.systems.slurm.slurm_system import SlurmSystem


class StubAgentConfig(BaseAgentConfig):
    knob: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class StubAgent(BaseAgent):
    received_configs: ClassVar[list[StubAgentConfig]] = []

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

    def select_action(self) -> tuple[int, dict[str, Any]]:
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
