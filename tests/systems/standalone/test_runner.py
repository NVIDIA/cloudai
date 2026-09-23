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

import cloudai.core
import cloudai.metrics
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
    job = StandaloneJob(
        test_run,
        id=123,
        start=start,
        finish=start + datetime.timedelta(seconds=3),
    )

    run = runner.get_run_output(job, test_run, runner.get_job_status(job))

    assert run.model_dump() == {
        "path": str((tmp_path / "case" / "0").absolute()),
        "jobid": "123",
        "status": "completed",
        "metrics": [
            {
                "name": "Bandwidth",
                "value": 12.5,
                "unit": "GB/s",
                "dimensions": [{"name": "Size", "value": "1024", "unit": "", "is_x": False}],
            }
        ],
        "start": start,
        "finish": start + datetime.timedelta(seconds=3),
        "duration": None,
        "iteration": 0,
        "step": 0,
    }
