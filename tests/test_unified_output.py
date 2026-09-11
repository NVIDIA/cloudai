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

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import toml

from cloudai._core.exceptions import JobFailureError
from cloudai.core import BaseJob, TestRun, TestScenario
from cloudai.systems.slurm import SlurmJob, SlurmRunner, SlurmSystem
from cloudai.unified_output import Experiment, ExperimentOutput
from cloudai.workloads.nccl_test import NCCLTestDefinition


@pytest.fixture
def runner(slurm_system: SlurmSystem, nccl_tr: TestRun, monkeypatch: pytest.MonkeyPatch) -> SlurmRunner:
    stdout = (nccl_tr.output_path / "stdout.txt").read_text() + "# Out of bounds values : 0\n"
    nccl_tr.iterations = 2
    slurm_system.output_path.mkdir()
    runner = SlurmRunner(
        "run", slurm_system, TestScenario("repeats", [nccl_tr]), slurm_system.output_path / "repeats_2026-09-11"
    )

    def submit(tr: TestRun) -> SlurmJob:
        content = stdout
        if tr.current_iteration:
            content = "\n".join(line for line in stdout.splitlines() if not line.lstrip().startswith("12000000"))
            content = content.replace("20.20", "40.20")
        (tr.output_path / "stdout.txt").write_text(content)
        return SlurmJob(tr, id=100 + tr.current_iteration)

    def complete(job: BaseJob) -> None:
        offset = "+02:00" if job.test_run.current_iteration == 0 else ""
        metadata = {
            "state": "COMPLETED",
            "exit_code": "0:0",
            "start_time": f"2026-09-11T12:00:00{offset}",
            "end_time": f"2026-09-11T12:00:02{offset}",
            "elapsed_time_sec": 2,
        }
        (job.test_run.output_path / "slurm-job.toml").write_text(toml.dumps(metadata))

    monkeypatch.setattr(runner, "on_job_submit", Mock())
    monkeypatch.setattr(runner, "_submit_test", submit)
    monkeypatch.setattr(runner, "on_job_completion", complete)
    monkeypatch.setattr(SlurmSystem, "is_job_completed", lambda self, job: True)
    monkeypatch.setattr(SlurmSystem, "is_job_running", lambda self, job: False)
    monkeypatch.setattr(SlurmSystem, "kill", Mock())
    return runner


def test_completed_experiment_and_repeated_metrics(runner: SlurmRunner):
    runner.run()

    path = runner.scenario_root / "experiment.json"
    experiment = Experiment.model_validate_json(path.read_text())
    data = json.loads(path.read_text())
    assert set(data) == {"id", "name", "status", "path", "start", "finish", "duration", "tests"}
    assert experiment.id == runner.scenario_root.name
    assert experiment.name == "repeats"
    assert experiment.status == "completed"
    assert experiment.start is not None and experiment.finish is not None
    assert experiment.duration == (experiment.finish - experiment.start).total_seconds()
    test = experiment.tests[0]
    assert test.id == test.name == "nccl_test"
    assert test.status == "completed"
    assert test.path == str(runner.scenario_root / "nccl_test")
    assert [run.iteration for run in test.runs] == [0, 1]
    assert [run.step for run in test.runs] == [0, 0]
    assert [run.jobid for run in test.runs] == ["100", "101"]
    assert [run.path for run in test.runs] == [str(Path(test.path) / str(i)) for i in range(2)]
    assert data["tests"][0]["runs"][0]["start"] == "2026-09-11T10:00:00Z"
    assert test.runs[1].start is None and test.runs[1].finish is None
    assert [run.duration for run in test.runs] == [2, 2]

    bandwidth = [
        metric
        for metric in test.metrics
        if metric.name == "Bandwidth"
        and any(dimension.name == "Placement" and dimension.value == "out_of_place" for dimension in metric.dimensions)
    ]
    assert [metric.value for metric in bandwidth] == pytest.approx([30.20, 30.30, 130.40])
    assert all(metric.unit == "GB/s" for metric in bandwidth)
    sizes = [dimension for metric in bandwidth for dimension in metric.dimensions if dimension.name == "Size"]
    assert [size.value for size in sizes] == ["1000000", "2000000", "12000000"]
    assert all(size.unit == "B" and size.is_x for size in sizes)
    assert len(test.runs[0].metrics) == 12
    assert len(test.runs[1].metrics) == 8
    assert test.runs[0].metrics != test.runs[1].metrics
    assert list(runner.scenario_root.glob("*.json")) == [path]


@pytest.mark.parametrize("failure,abort", [("workload", True), ("workload", False), ("scheduler", False)])
def test_failed_run_preserves_earlier_results(
    runner: SlurmRunner, monkeypatch: pytest.MonkeyPatch, failure: str, abort: bool
):
    complete = runner.on_job_completion
    runner.test_scenario.job_status_check = abort

    def fail_second_job(job: BaseJob) -> None:
        complete(job)
        if job.test_run.current_iteration != 1:
            return
        if failure == "workload":
            with (job.test_run.output_path / "stdout.txt").open("a") as stdout:
                stdout.write("\nTest NCCL failure\n")
        else:
            path = job.test_run.output_path / "slurm-job.toml"
            path.write_text(path.read_text().replace("COMPLETED", "FAILED"))

    monkeypatch.setattr(runner, "on_job_completion", fail_second_job)
    if abort:
        with pytest.raises(JobFailureError):
            runner.run()
    else:
        runner.run()

    experiment = Experiment.model_validate_json((runner.scenario_root / "experiment.json").read_text())
    test = experiment.tests[0]
    assert experiment.status == test.status == "failed"
    assert [run.status for run in test.runs] == ["completed", "failed"]
    assert [run.iteration for run in test.runs] == [0, 1]
    assert test.runs[0].path != test.runs[1].path
    assert len(test.runs[1].metrics) == 8
    assert test.metrics == test.runs[0].metrics


def test_output_failures_are_nonfatal(
    runner: SlurmRunner, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    extract = NCCLTestDefinition.metric_observations

    def failing_extract(self, system, tr):
        if tr.current_iteration == 1:
            raise ValueError("malformed measurement")
        return extract(self, system, tr)

    monkeypatch.setattr(NCCLTestDefinition, "metric_observations", failing_extract)
    runner.run()

    path = runner.scenario_root / "experiment.json"
    original = path.read_bytes()
    experiment = Experiment.model_validate_json(original)
    assert experiment.status == "completed"
    assert experiment.tests[0].runs[1].metrics == []
    assert experiment.tests[0].metrics == experiment.tests[0].runs[0].metrics
    assert "Cannot extract unified output metrics for job 101: malformed measurement" in caplog.text

    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("write unavailable")))
    output = ExperimentOutput(runner.test_scenario, runner.scenario_root)
    output.finish(runner.system, [], completed=False)
    assert path.read_bytes() == original
    assert "Cannot write unified experiment output: write unavailable" in caplog.text
    assert sorted(p.name for p in runner.scenario_root.iterdir()) == ["experiment.json", "nccl_test"]

    monkeypatch.setattr(
        "cloudai.systems.slurm.slurm_runner.ExperimentOutput", Mock(side_effect=OSError("initialization unavailable"))
    )
    runner.test_scenario.test_runs[0].current_iteration = 0
    runner.run()
    assert len(runner.jobs) == 0
    assert path.read_bytes() == original
    assert "Cannot initialize unified experiment output: initialization unavailable" in caplog.text
