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

import argparse
import datetime
import pathlib
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

import cloudai.core
import cloudai.handlers
import cloudai.metrics
import cloudai.models.output
from cloudai.cli.cli import handle_dry_run_and_run
from cloudai.handlers import handle_dse_job
from cloudai.systems.standalone import StandaloneJob, StandaloneRunner, StandaloneSystem
from tests.configurator import probe
from tests.configurator.probe import dse_runner, probe_test_run, probe_timing, read_trajectory


class MetricWorkload(cloudai.core.TestDefinition):
    def was_run_successful(self, tr: cloudai.core.TestRun) -> cloudai.core.JobStatusResult:
        return cloudai.core.JobStatusResult(is_successful=True)

    def metric_observations(
        self, system: cloudai.core.System, tr: cloudai.core.TestRun
    ) -> list[cloudai.metrics.MetricObservation]:
        return [
            cloudai.metrics.MetricObservation(
                metric=cloudai.metrics.BANDWIDTH,
                value=12.5,
                dimensions={"size_bytes": 1024},
            )
        ]


def test_standalone_run_output_uses_workload_status_and_metrics(
    tmp_path: pathlib.Path, standalone_system: StandaloneSystem
) -> None:
    workload = MetricWorkload(
        name="metric-workload",
        description="metric workload",
        test_template_name="MetricWorkload",
        cmd_args=cloudai.core.CmdArgs(),
    )
    test_run = cloudai.core.TestRun(
        name="case",
        test=workload,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "case" / "0",
    )
    runner = StandaloneRunner(
        "run", standalone_system, cloudai.core.TestScenario(name="scenario", test_runs=[test_run]), tmp_path
    )
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    job = StandaloneJob(test_run, id=123, start=start)
    runner.jobs.append(job)
    runner.testrun_to_job_map[test_run] = job
    runner.update_run_output(job)
    initial = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    datetime_type = datetime.datetime
    with (
        mock.patch.object(StandaloneSystem, "is_job_running", return_value=True),
        mock.patch.object(StandaloneSystem, "is_job_completed", side_effect=[False, False, True]),
        mock.patch("cloudai.output.datetime.datetime", wraps=datetime_type) as clock,
        mock.patch.object(runner.experiment_output, "write", wraps=runner.experiment_output.write) as write,
    ):
        for seconds, expected in ((1, "running"), (2, "running"), (3, "completed")):
            write.reset_mock()
            clock.now.return_value = start + datetime.timedelta(seconds=seconds)
            runner.monitor_jobs()
            stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
            assert len(stored.tests[0].runs) == 1
            run = stored.tests[0].runs[0]
            assert stored.tests[0].status == expected
            assert run.status == expected
            if expected == "running":
                write.assert_not_called()
                assert stored == initial
            else:
                write.assert_called_once()
                assert run.duration == seconds

    stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    run = stored.tests[0].runs[0]
    assert run.model_dump() == {
        "path": str((tmp_path / "case" / "0").absolute()),
        "jobid": "123",
        "status": "completed",
        "metrics": [
            {
                "name": "Bandwidth",
                "value": 12.5,
                "unit": "GB/s",
                "dimensions": [{"name": "Size", "value": "1024", "unit": "", "is_x": True}],
            }
        ],
        "start": start,
        "finish": start + datetime.timedelta(seconds=3),
        "duration": 3,
        "iteration": 0,
        "step": 0,
    }
    runner.experiment_output.finish("completed", job.finish)
    stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert stored.status == "completed"
    assert stored.tests[0].metrics == run.metrics


@pytest.fixture
def standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StandaloneSystem:
    monkeypatch.setitem(
        cloudai.core.Registry().command_gen_strategies_map,
        (StandaloneSystem, probe.ProbeTestDefinition),
        probe.ProbeStandaloneCommandGenStrategy,
    )
    return StandaloneSystem(
        name="standalone", install_path=tmp_path / "install", output_path=tmp_path / "output", monitor_interval=0.05
    )


def test_free_slots_are_refilled_while_a_slow_process_runs(standalone: StandaloneSystem) -> None:
    test_run = probe_test_run(
        [1, 2, 3, 4, 5, 6],
        agent_config={"max_parallelism": 3},
        duration=0.4,
        durations={1: 2.5},
        rejected_x=[2],
        failing_x=[4],
    )
    runner = dse_runner(standalone, StandaloneRunner, test_run)

    assert handle_dse_job(runner) == 0

    assert sorted(read_trajectory(runner)) == [(1, 1, 1.0), (2, 3, 3.0), (3, 4, -1.0), (4, 5, 5.0), (5, 6, 6.0)]
    trial_root = runner.runner.scenario_root / "probe" / "0"
    timing = {step: probe_timing(trial_root / str(step)) for step in range(1, 6)}
    slow_start, slow_finish = timing[1]
    # The slow trial holds one slot for the whole sweep; the other two slots are refilled meanwhile.
    assert all(start < slow_finish and slow_start < finish for start, finish in timing.values())
    assert min(timing[4][0], timing[5][0]) >= min(timing[2][1], timing[3][1])
    changes = sorted([(start, 1) for start, _ in timing.values()] + [(finish, -1) for _, finish in timing.values()])
    alive = peak = 0
    for _, change in changes:
        alive += change
        peak = max(peak, alive)
    assert peak == 3

    runs = runner.runner.experiment_output.experiment.tests[0].runs
    assert [(run.step, run.status) for run in runs] == [
        (1, "completed"),
        (2, "completed"),
        (3, "failed"),
        (4, "completed"),
        (5, "completed"),
    ]


def test_cancelling_kills_the_running_processes(standalone: StandaloneSystem, caplog: pytest.LogCaptureFixture) -> None:
    test_run = probe_test_run([1, 2, 3, 4], agent_config={"max_parallelism": 2}, duration=30)
    next_case = probe_test_run([8, 9], name="next_case")
    runner = dse_runner(standalone, StandaloneRunner, test_run, next_case)
    processes = []
    submit = runner.runner.submit_test

    def submit_and_remember(tr) -> None:
        submit(tr)
        job = runner.runner.jobs[-1]
        if isinstance(job, StandaloneJob):
            processes.append(job.process)

    runner.runner.submit_test = submit_and_remember
    canceller = threading.Timer(0.5, runner.runner.shutdown)
    canceller.start()
    started = time.monotonic()

    try:
        assert handle_dse_job(runner) == 1
    finally:
        canceller.cancel()

    assert time.monotonic() - started < 15
    assert len(processes) == 2
    assert all(process is not None and process.poll() is not None for process in processes)
    assert read_trajectory(runner) == []
    assert not (runner.runner.scenario_root / "next_case").exists()
    runner.runner.finish_output(successful=False)
    assert runner.runner.experiment_output.experiment.status == "cancelled"
    assert all(run.status == "cancelled" for run in runner.runner.experiment_output.experiment.tests[0].runs)
    assert "Step 1 failed" not in caplog.text
    assert "Step 2 failed" not in caplog.text


@pytest.mark.parametrize("job_status_check", [True, False])
def test_cli_reports_ordinary_scenario_failure_policy(
    standalone: StandaloneSystem, monkeypatch: pytest.MonkeyPatch, job_status_check: bool
) -> None:
    test_run = probe_test_run([1], failing_x=[1], duration=0.05)
    test_run.test.cmd_args.x = 1
    runner = dse_runner(standalone, StandaloneRunner, test_run)
    runner.runner.test_scenario.job_status_check = job_status_check
    monkeypatch.setattr(
        cloudai.handlers, "load_experiment", lambda *args, **kwargs: (standalone, [], runner.runner.test_scenario)
    )
    monkeypatch.setattr(cloudai.handlers, "create_experiment_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(cloudai.handlers, "_ensure_installation", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        test_scenario=Path("scenario.toml"),
        system_config=Path("system.toml"),
        tests_dir=None,
        hook_dir=None,
        output_dir=None,
        mode="run",
        single_sbatch=False,
        enable_cache_without_check=False,
    )

    assert handle_dry_run_and_run(args) == (1 if job_status_check else 0)
    assert (runner.runner.scenario_root / "probe" / "0" / "probe_status.txt").read_text() == "failed"
