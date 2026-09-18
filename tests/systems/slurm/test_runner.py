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
from cloudai.systems.slurm import SlurmJob, SlurmRunner, SlurmSystem
from cloudai.systems.slurm.slurm_metadata import SlurmStepMetadata


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
    )
    observation = cloudai.metrics.MetricObservation(cloudai.metrics.BANDWIDTH, 12.5, {"size_bytes": 1024})
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
                    "dimensions": [{"name": "Size", "value": "1024", "unit": "", "is_x": False}],
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
