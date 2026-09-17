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

import pytest

import cloudai.metrics
from cloudai.core import JobStatusResult, System, TestDefinition, TestRun, TestScenario
from cloudai.models.workload import CmdArgs
from cloudai.systems.standalone import StandaloneJob, StandaloneRunner, StandaloneSystem


class MetricWorkload(TestDefinition):
    successful: bool

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        if self.successful:
            return JobStatusResult(is_successful=True)
        return JobStatusResult(is_successful=False, error_message="workload result failed")

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        return [
            cloudai.metrics.MetricObservation(
                metric=cloudai.metrics.BANDWIDTH,
                value=12.5,
                dimensions={"size_bytes": 1024},
            )
        ]


@pytest.mark.parametrize("successful", [True, False])
def test_standalone_run_output_uses_workload_status_and_metrics(
    tmp_path: pathlib.Path, standalone_system: StandaloneSystem, successful: bool
) -> None:
    workload = MetricWorkload(
        name="metric-workload",
        description="metric workload",
        test_template_name="MetricWorkload",
        cmd_args=CmdArgs(),
        successful=successful,
    )
    test_run = TestRun(
        name="case",
        test=workload,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "case" / "0",
    )
    runner = StandaloneRunner("run", standalone_system, TestScenario(name="scenario", test_runs=[test_run]), tmp_path)
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    job = StandaloneJob(
        test_run,
        id=123,
        start=start,
        finish=start + datetime.timedelta(seconds=3),
    )

    run = runner.get_run_output(job, test_run, runner.get_job_status(job))

    assert run.status == ("completed" if successful else "failed")
    assert run.jobid == "123"
    assert run.start == start
    assert run.finish == start + datetime.timedelta(seconds=3)
    if successful:
        assert [metric.model_dump() for metric in run.metrics] == [
            {
                "name": "Bandwidth",
                "value": 12.5,
                "unit": "GB/s",
                "dimensions": [{"name": "Size", "value": "1024", "unit": "", "is_x": False}],
            }
        ]
    else:
        assert run.metrics == []
