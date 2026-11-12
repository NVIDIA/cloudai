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

"""Parallel DSE with real processes on the standalone system."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from cloudai.handlers import handle_dse_job
from cloudai.systems.standalone import StandaloneJob, StandaloneRunner, StandaloneSystem

from .probe import dse_runner, probe_test_run, probe_timing, read_trajectory


@pytest.fixture
def standalone(tmp_path: Path) -> StandaloneSystem:
    return StandaloneSystem(
        name="standalone", install_path=tmp_path / "install", output_path=tmp_path / "output", monitor_interval=0.05
    )


def test_processes_of_a_wave_overlap_and_waves_do_not(standalone: StandaloneSystem) -> None:
    test_run = probe_test_run(
        [1, 2, 3, 4, 5, 6],
        agent_config={"max_parallelism": 3},
        duration=0.7,
        rejected_x=[2],
        failing_x=[4],
    )
    runner = dse_runner(standalone, StandaloneRunner, test_run)

    assert handle_dse_job(runner) == 0

    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 3, 3.0), (3, 4, -1.0), (4, 5, 5.0), (5, 6, 6.0)]
    trial_root = runner.runner.scenario_root / "probe" / "0"
    timing = {step: probe_timing(trial_root / str(step)) for step in range(1, 6)}
    first_wave, second_wave = [timing[step] for step in (1, 2, 3)], [timing[step] for step in (4, 5)]
    for wave in (first_wave, second_wave):
        assert max(start for start, _ in wave) < min(finish for _, finish in wave)
    assert min(start for start, _ in second_wave) >= max(finish for _, finish in first_wave)

    runs = runner.runner.experiment_output.experiment.tests[0].runs
    assert [(run.step, run.status) for run in runs] == [
        (1, "completed"),
        (2, "completed"),
        (3, "failed"),
        (4, "completed"),
        (5, "completed"),
    ]


def test_cancelling_kills_the_running_processes(standalone: StandaloneSystem) -> None:
    test_run = probe_test_run([1, 2, 3, 4], agent_config={"max_parallelism": 2}, duration=30)
    runner = dse_runner(standalone, StandaloneRunner, test_run)
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
