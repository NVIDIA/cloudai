# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Probe workload and runners for DSE tests.

The probe workload has one swept argument, ``x``. What happens to a trial is decided by ``x``
alone, so a test reads as a list of proposals and their fates:

- ``rejected_x``: the constraint check fails, nothing is executed;
- ``failing_x``: the trial is executed, fails, and reports no metric;
- ``durations``: how long the trial runs; everything else takes ``duration``.

A finished trial reports ``x`` as its ``default`` metric. Two ways to execute it:

- ``ScriptedRunner`` with ``ScriptedSystem``: jobs live in memory and durations are counted in
  monitoring passes, so wave tests are exact and need no waiting;
- ``StandaloneRunner`` with ``ProbeStandaloneCommandGenStrategy``: real processes, durations in
  seconds, each process records when it started and finished.
"""

from __future__ import annotations

import ast
import csv
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pydantic import Field, PrivateAttr

from cloudai.core import (
    METRIC_ERROR,
    BaseJob,
    BaseRunner,
    CommandGenStrategy,
    Installable,
    JobStatusResult,
    JobSubmissionError,
    MetricValue,
    ReportGenerationStrategy,
    Runner,
    System,
    TestRun,
    TestScenario,
)
from cloudai.models.workload import CmdArgs, TestDefinition

RESULT_FILE = "probe_result.txt"
TIMING_FILE = "probe_timing.json"
JOB_SCRIPT = Path(__file__).with_name("probe_job.py")


class ProbeCmdArgs(CmdArgs):
    """Probe arguments; ``x`` is the swept one."""

    x: int | list[int]
    noise: int | list[int] = 0


class ProbeTestDefinition(TestDefinition):
    """Workload whose trial behaviour is fully decided by ``x``."""

    cmd_args: ProbeCmdArgs
    rejected_x: list[int] = Field(default_factory=list)
    failing_x: list[int] = Field(default_factory=list)
    unsubmittable_x: list[int] = Field(default_factory=list)
    durations: dict[int, float] = Field(default_factory=dict)
    duration: float = 1

    @property
    def installables(self) -> list[Installable]:
        return []

    def x_of(self, tr: TestRun) -> int:
        x = tr.test.cmd_args.x
        if isinstance(x, list):
            raise TypeError("x is not resolved to a single value")
        return x

    def duration_of(self, tr: TestRun) -> float:
        return self.durations.get(self.x_of(tr), self.duration)

    def constraint_check(self, tr: TestRun, system: Optional[System]) -> bool:
        return self.x_of(tr) not in self.rejected_x

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        if (tr.output_path / RESULT_FILE).exists():
            return JobStatusResult(is_successful=True)
        return JobStatusResult(is_successful=False, error_message=f"probe x={self.x_of(tr)} failed")


class ProbeReport(ReportGenerationStrategy):
    """Reports the value a finished probe trial wrote."""

    def can_handle_directory(self) -> bool:
        return (self.test_run.output_path / RESULT_FILE).exists()

    def generate_report(self) -> None:
        return

    def get_metric(self, metric: str) -> MetricValue:
        if metric != "default" or not self.can_handle_directory():
            return METRIC_ERROR
        return float((self.test_run.output_path / RESULT_FILE).read_text())


def probe_test_run(
    x: list[int],
    *,
    agent: str = "grid_search",
    agent_steps: int = 1,
    agent_config: dict | None = None,
    name: str = "probe",
    **behaviour,
) -> TestRun:
    """Build a DSE test run over ``x``; ``behaviour`` sets the ``ProbeTestDefinition`` fields."""
    test = ProbeTestDefinition(
        name=name,
        description="probe",
        test_template_name="Probe",
        cmd_args=ProbeCmdArgs(x=x),
        agent=agent,
        agent_steps=agent_steps,
        agent_config=agent_config,
        agent_reward_function="identity",
        **behaviour,
    )
    return TestRun(name=name, test=test, num_nodes=1, nodes=[], reports={ProbeReport})


@dataclass
class ScriptedJob(BaseJob):
    """In-memory job that finishes after a number of monitoring passes."""

    remaining_passes: float = 0
    killed: bool = False
    finished: bool = False


class ScriptedSystem(System):
    """System whose jobs live in memory; ``events`` records what happened to each trial."""

    scheduler: str = "scripted"
    monitor_interval: float = 0
    _events: list[tuple[str, int]] = PrivateAttr(default_factory=list)
    _on_pass: Callable[[], None] | None = PrivateAttr(default=None)

    @property
    def events(self) -> list[tuple[str, int]]:
        """``(event, trial step)`` in the order they happened: submit, finish, fail, kill."""
        return self._events

    def on_pass(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` at every completion check, e.g. to cancel the run midway."""
        self._on_pass = callback

    def record(self, event: str, tr: TestRun) -> None:
        self._events.append((event, tr.step))

    def update(self) -> None:
        return

    def is_job_running(self, job: BaseJob) -> bool:
        return isinstance(job, ScriptedJob) and not job.finished

    def is_job_completed(self, job: BaseJob) -> bool:
        if not isinstance(job, ScriptedJob):
            raise TypeError("ScriptedSystem runs only scripted jobs")
        if self._on_pass is not None:
            self._on_pass()
        if job.finished:
            return True

        job.remaining_passes -= 1
        if job.remaining_passes > 0 and not job.killed:
            return False

        job.finished = True
        tr = job.test_run
        test = tr.test
        if not isinstance(test, ProbeTestDefinition):
            raise TypeError("ScriptedSystem runs only the probe workload")
        if job.killed:
            return True
        if test.x_of(tr) in test.failing_x:
            self.record("fail", tr)
        else:
            (tr.output_path / RESULT_FILE).write_text(str(test.x_of(tr)))
            self.record("finish", tr)
        return True

    def kill(self, job: BaseJob) -> None:
        if isinstance(job, ScriptedJob) and not job.finished and not job.killed:
            job.killed = True
            self.record("kill", job.test_run)


class ScriptedRunner(BaseRunner):
    """Runner for ``ScriptedSystem``: real submission and polling logic, in-memory jobs."""

    def _submit_test(self, tr: TestRun) -> BaseJob:
        test = tr.test
        if not isinstance(test, ProbeTestDefinition) or not isinstance(self.system, ScriptedSystem):
            raise TypeError("ScriptedRunner runs only the probe workload on a ScriptedSystem")
        if test.x_of(tr) in test.unsubmittable_x:
            raise JobSubmissionError(tr.name, "probe", "", "", "scripted submission failure")
        self.system.record("submit", tr)
        return ScriptedJob(tr, id=tr.step, remaining_passes=test.duration_of(tr))


def scripted_system(tmp_path: Path) -> ScriptedSystem:
    return ScriptedSystem(name="scripted", install_path=tmp_path / "install", output_path=tmp_path / "output")


def dse_runner(system: System, runner_class: type[BaseRunner], *test_runs: TestRun) -> Runner:
    """Build the runner ``handle_dse_job`` takes, in ``run`` mode."""
    system.output_path.mkdir(parents=True, exist_ok=True)
    scenario = TestScenario(name="probe_scenario", test_runs=list(test_runs))
    return Runner(mode="run", system=system, test_scenario=scenario, runner_class=runner_class)


def read_trajectory(runner: Runner, name: str = "probe") -> list[tuple[int, int, float]]:
    """Return ``(step, x, reward)`` for every trajectory row of a test run."""
    path = runner.runner.scenario_root / name / "0" / "trajectory.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            (int(row["step"]), ast.literal_eval(row["action"])["x"], float(row["reward"])) for row in csv.DictReader(f)
        ]


def max_outstanding(events: list[tuple[str, int]]) -> int:
    """Return the largest number of trials that were submitted and not yet over at the same time."""
    outstanding = peak = 0
    for event, _ in events:
        outstanding += 1 if event == "submit" else -1
        peak = max(peak, outstanding)
    return peak


def waves(events: list[tuple[str, int]]) -> list[list[int]]:
    """Group submitted trial steps into waves: a wave ends when the next submit follows a completion."""
    result: list[list[int]] = []
    previous = None
    for event, step in events:
        if event == "submit":
            if previous != "submit":
                result.append([])
            result[-1].append(step)
        previous = event
    return result


class ProbeStandaloneCommandGenStrategy(CommandGenStrategy):
    """Runs a probe trial as a real process that records its own start and finish time."""

    def store_test_run(self) -> None:
        return

    def gen_exec_command(self) -> str:
        tr = self.test_run
        test = tr.test
        if not isinstance(test, ProbeTestDefinition):
            raise TypeError("ProbeStandaloneCommandGenStrategy runs only the probe workload")
        args = [sys.executable, str(JOB_SCRIPT), str(tr.output_path), str(test.x_of(tr)), str(test.duration_of(tr))]
        if test.x_of(tr) in test.failing_x:
            args.append("--fail")
        return " ".join(shlex.quote(arg) for arg in args)


def probe_timing(output_path: Path) -> tuple[float, float]:
    """Return when the process of a real probe trial started and finished."""
    timing = json.loads((output_path / TIMING_FILE).read_text())
    return timing["start"], timing["finish"]
