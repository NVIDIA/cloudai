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

import datetime
import pathlib
from unittest import mock

import pytest

import cloudai.core
import cloudai.metrics
import cloudai.models.output
from cloudai.systems.slurm import SlurmJob, SlurmRunner, SlurmSystem
from cloudai.systems.slurm.slurm_metadata import SlurmStepMetadata
from cloudai.systems.slurm.slurm_rest_client import SlurmAPIConfig, SlurmRestClient


@pytest.mark.parametrize(
    "successful,state,status",
    [(True, "FAILED", "completed"), (False, "COMPLETED", "failed"), (False, "CANCELLED", "cancelled")],
)
@pytest.mark.parametrize("step", [0, 2])
def test_slurm_run_output(
    tmp_path: pathlib.Path,
    base_tr: cloudai.core.TestRun,
    slurm_system: SlurmSystem,
    successful: bool,
    state: str,
    status: str,
    step: int,
) -> None:
    runner = SlurmRunner("run", slurm_system, cloudai.core.TestScenario(name="scenario", test_runs=[base_tr]), tmp_path)
    base_tr.output_path.mkdir(parents=True)
    base_tr.step = step
    job = SlurmJob(base_tr, id=123)
    runner.update_run_output(job)
    metadata = SlurmStepMetadata(
        job_id=123,
        step_id="",
        name="job",
        state=state,
        exit_code="1:0",
        elapsed_time_sec=3,
        start_time="2026-01-02T03:04:05Z",
        end_time="2026-01-02T03:04:08Z",
        submit_line="sbatch run.sh",
        cluster_name="actual-cluster",
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.BANDWIDTH,
        12.5,
        {"size_bytes": 1024, "bandwidth_basis": "bus"},
    )
    with (
        mock.patch.object(SlurmSystem, "get_job_status", return_value=[metadata]) as get_metadata,
        mock.patch.object(
            runner,
            "get_cmd_gen_strategy",
            return_value=mock.Mock(gen_srun_command=lambda: "srun cmd", generate_test_command=lambda: ["cmd"]),
        ),
        mock.patch.object(
            cloudai.core.TestDefinition,
            "was_run_successful",
            return_value=cloudai.core.JobStatusResult(is_successful=successful),
        ),
        mock.patch.object(
            cloudai.core.TestDefinition, "metric_observations", return_value=[observation]
        ) as get_metrics,
    ):
        runner.store_job_metadata(job)
        runner.update_run_output(job, runner.get_job_status(job))
        get_metadata.assert_called_once_with(job)
        assert get_metrics.call_count == int(successful)

    base_tr.step = 3
    runner.shutting_down = status == "cancelled"
    runner.finish_output(successful=True)
    experiment = runner.experiment_output.snapshot()
    assert experiment.status == status
    assert experiment.system_name == "actual-cluster"
    test = experiment.tests[0]
    assert test.metrics == (test.runs[0].metrics if successful and step == 0 else [])
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    assert [run.model_dump() for run in test.runs] == [
        {
            "path": str(base_tr.output_path.absolute()),
            "jobid": "123",
            "status": status,
            "metrics": [
                {
                    "name": "Bandwidth",
                    "value": 12.5,
                    "unit": "GB/s",
                    "dimensions": [
                        {"name": "Bandwidth basis", "value": "bus", "unit": "", "is_x": False},
                        {"name": "Size", "value": "1024", "unit": "", "is_x": True},
                    ],
                }
            ]
            if successful
            else [],
            "start": start,
            "finish": start + datetime.timedelta(seconds=3),
            "duration": 3,
            "iteration": 0,
            "step": step,
        }
    ]


@pytest.mark.parametrize("timestamp", ["Unknown", "", "2026-01-02T03:04:05"])
def test_slurm_output_unknown_timing_and_metric_failure(
    tmp_path: pathlib.Path,
    base_tr: cloudai.core.TestRun,
    slurm_system: SlurmSystem,
    timestamp: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = SlurmRunner("run", slurm_system, cloudai.core.TestScenario(name="scenario", test_runs=[base_tr]), tmp_path)
    job = SlurmJob(base_tr, id=123)
    with mock.patch.object(SlurmSystem, "get_job_status", return_value=[]):
        runner.store_job_metadata(job)
    with mock.patch.object(
        cloudai.core.TestDefinition, "metric_observations", side_effect=ValueError("broken metrics")
    ):
        run = runner.get_run_output(job, base_tr, cloudai.core.JobStatusResult(is_successful=True))
    assert run is not None
    assert run.model_dump() == {
        "path": str(base_tr.output_path.absolute()),
        "jobid": "123",
        "status": "completed",
        "metrics": [],
        "start": None,
        "finish": None,
        "duration": None,
        "iteration": 0,
        "step": 0,
    }
    assert runner._output_timestamp(timestamp) is None
    assert "broken metrics" in caplog.text


@pytest.mark.parametrize(
    "state,status",
    [
        ("PENDING", "pending"),
        ("CONFIGURING", "pending"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("SUSPENDED", "running"),
        ("UNRECOGNIZED", "unknown"),
    ],
)
def test_slurm_live_output(
    tmp_path: pathlib.Path, base_tr: cloudai.core.TestRun, slurm_system: SlurmSystem, state: str, status: str
) -> None:
    runner = SlurmRunner("run", slurm_system, cloudai.core.TestScenario(name="scenario", test_runs=[base_tr]), tmp_path)
    job = SlurmJob(base_tr, id=123)
    runner.jobs.append(job)
    runner.update_run_output(job)
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    metadata = SlurmStepMetadata(
        job_id=123,
        step_id="",
        name="job",
        state=state,
        exit_code="0:0",
        start_time=start.isoformat(),
        end_time="Unknown",
        elapsed_time_sec=0,
        submit_line="sbatch run.sh",
        cluster_name="actual-cluster",
    )
    # Job steps must not override the allocation's state, regardless of ordering.
    step = metadata.model_copy(update={"step_id": "batch", "state": "RUNNING"})
    with (
        mock.patch.object(SlurmSystem, "is_job_completed", return_value=False),
        mock.patch.object(SlurmSystem, "get_job_status", return_value=[step, metadata]) as get_metadata,
        mock.patch.object(cloudai.core.TestDefinition, "metric_observations") as get_metrics,
    ):
        assert runner.monitor_jobs() == 0
        get_metadata.assert_called_once_with(job)
        get_metrics.assert_not_called()

    experiment = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert experiment.status == "running"
    assert experiment.system_name == "actual-cluster"
    assert experiment.duration is not None
    test = experiment.tests[0]
    assert test.status == status
    assert len(test.runs) == 1
    run = test.runs[0]
    assert run.status == status
    assert run.jobid == "123"
    assert run.path == str(base_tr.output_path.absolute())
    assert run.finish is None
    assert run.metrics == []
    if status == "running":
        assert run.start == start
        assert run.duration is not None and run.duration >= 10
    else:
        assert run.start is None
        assert run.duration is None
    assert job.metadata is None


@pytest.mark.parametrize("unavailable", [[], RuntimeError("accounting unavailable"), ValueError("invalid metadata")])
def test_slurm_live_output_retains_last_observation(
    tmp_path: pathlib.Path, base_tr: cloudai.core.TestRun, slurm_system: SlurmSystem, unavailable: object
) -> None:
    runner = SlurmRunner("run", slurm_system, cloudai.core.TestScenario(name="scenario", test_runs=[base_tr]), tmp_path)
    job = SlurmJob(base_tr, id=123)
    runner.jobs.append(job)
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    pending = SlurmStepMetadata(
        job_id=123,
        step_id="",
        name="job",
        state="PENDING",
        exit_code="0:0",
        start_time="Unknown",
        end_time="Unknown",
        elapsed_time_sec=0,
        submit_line="sbatch run.sh",
    )
    running = pending.model_copy(update={"state": "RUNNING", "start_time": start.isoformat()})
    completed = running.model_copy(
        update={
            "state": "COMPLETED",
            "end_time": (start + datetime.timedelta(seconds=10)).isoformat(),
            "elapsed_time_sec": 10,
        }
    )
    with (
        mock.patch.object(SlurmSystem, "is_job_completed", side_effect=[False, False, False, True]),
        mock.patch.object(SlurmSystem, "get_job_status", side_effect=[[pending], [running], unavailable, [completed]]),
        mock.patch.object(SlurmSystem, "complete_job", return_value=[]),
        mock.patch.object(
            runner,
            "get_cmd_gen_strategy",
            return_value=mock.Mock(gen_srun_command=lambda: "srun cmd", generate_test_command=lambda: ["cmd"]),
        ),
        mock.patch.object(
            cloudai.core.TestDefinition,
            "was_run_successful",
            return_value=cloudai.core.JobStatusResult(is_successful=True),
        ),
    ):
        runner.testrun_to_job_map[base_tr] = job
        base_tr.output_path.mkdir(parents=True)
        for status in ("pending", "running", "running", "completed"):
            runner.monitor_jobs()
            experiment = cloudai.models.output.Experiment.model_validate_json(
                (tmp_path / "experiment.json").read_text()
            )
            assert experiment.tests[0].status == status
            assert len(experiment.tests[0].runs) == 1
            assert experiment.tests[0].runs[0].status == status
    experiment = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    run = experiment.tests[0].runs[0]
    assert run.start == start
    assert run.finish == start + datetime.timedelta(seconds=10)
    assert run.duration == 10
    assert runner.jobs == []


@pytest.mark.parametrize("backend", ["cli", "rest"])
def test_slurm_live_output_uses_transport(
    tmp_path: pathlib.Path, base_tr: cloudai.core.TestRun, slurm_system: SlurmSystem, backend: str
) -> None:
    if backend == "rest":
        slurm_system.slurm_api = SlurmAPIConfig(url="https://slurm.example.com")
    runner = SlurmRunner("run", slurm_system, cloudai.core.TestScenario(name="scenario", test_runs=[base_tr]), tmp_path)
    job = SlurmJob(base_tr, id=123)
    runner.jobs.append(job)
    start = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) - datetime.timedelta(seconds=10)
    stdout = f"123|job|RUNNING|0:0|{start.isoformat()}|Unknown|10|actual-cluster|sbatch run.sh|\n"
    process = mock.Mock()
    process.communicate.return_value = (stdout, "")
    response = {
        "meta": {"slurm": {"cluster": "actual-cluster"}},
        "jobs": [
            {
                "job_id": 123,
                "name": "job",
                "job_state": "RUNNING",
                "exit_code": 0,
                "start_time": int(start.timestamp()),
                "end_time": 0,
            }
        ],
    }
    with (
        mock.patch.object(SlurmSystem, "is_job_completed", return_value=False),
        mock.patch.object(slurm_system.cmd_shell, "execute", return_value=process) as execute,
        mock.patch.object(SlurmRestClient, "_request", return_value=response) as request,
    ):
        runner.monitor_jobs()
        if backend == "rest":
            request.assert_called_once_with("GET", "slurm", "job/123", retry_threshold=3)
            execute.assert_not_called()
        else:
            assert "sacct -j 123" in execute.call_args.args[0]
            request.assert_not_called()
    experiment = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert experiment.system_name == "actual-cluster"
    assert experiment.tests[0].status == "running"
    run = experiment.tests[0].runs[0]
    assert run.status == "running"
    assert run.start == start
    assert run.finish is None
    assert run.duration is not None and run.duration >= 10
