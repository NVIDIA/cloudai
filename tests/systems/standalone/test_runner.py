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

import cloudai.core
import cloudai.metrics
import cloudai.models.output
from cloudai.systems.standalone import StandaloneJob, StandaloneRunner, StandaloneSystem


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
        mock.patch.object(StandaloneSystem, "is_job_completed", side_effect=[False, False, False, False, True, True]),
        mock.patch("cloudai.output.datetime.datetime", wraps=datetime_type) as clock,
        mock.patch.object(runner.experiment_output, "write", wraps=runner.experiment_output.write) as write,
    ):
        for seconds, expected in ((1, "running"), (2, "running"), (3, "completed")):
            write.reset_mock()
            clock.now.return_value = start + datetime.timedelta(seconds=seconds)
            runner.check_start_post_init_dependencies()
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
