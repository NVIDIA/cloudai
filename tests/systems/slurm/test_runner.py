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
    runner = SlurmRunner(
        "run",
        slurm_system,
        cloudai.core.TestScenario(name="scenario", test_runs=[base_tr], job_status_check=False),
        tmp_path,
    )
    base_tr.output_path.mkdir(parents=True)
    base_tr.step = step
    job = SlurmJob(base_tr, id=123)
    runner.jobs.append(job)
    runner.testrun_to_job_map[base_tr] = job
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
        mock.patch.object(SlurmSystem, "is_job_running", side_effect=[False, True, False]) as is_running,
        mock.patch.object(
            SlurmSystem, "is_job_completed", side_effect=[False, False, False, False, True, True]
        ) as is_completed,
        mock.patch.object(SlurmSystem, "complete_job", return_value=[]),
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
        for expected in ("pending", "running", status):
            is_running.reset_mock()
            is_completed.reset_mock()
            runner.check_start_post_init_dependencies()
            runner.monitor_jobs()
            is_running.assert_called_once_with(job)
            assert is_completed.call_count == 2
            stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
            assert stored.tests[0].status == expected
            assert len(stored.tests[0].runs) == 1
            run = stored.tests[0].runs[0]
            assert run.status == expected
            if expected in ("pending", "running"):
                assert run.start is None and run.finish is None and run.duration is None
                get_metadata.assert_not_called()
                get_metrics.assert_not_called()
        get_metadata.assert_called_once_with(job)
        assert get_metrics.call_count == int(successful)

    base_tr.step = 3
    runner.shutting_down = status == "cancelled"
    runner.finish_output(successful=True)
    experiment = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
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
