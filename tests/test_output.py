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

import cloudai.models.output
import cloudai.output


def test_experiment_output_preserves_runs_and_finalizes_failure(tmp_path: pathlib.Path) -> None:
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    experiment = cloudai.models.output.Experiment(
        id="experiment",
        name="scenario",
        status="running",
        path=str(tmp_path),
        start=start,
        tests=[cloudai.models.output.Test(id="case", name="workload", path=str(tmp_path / "case"))],
    )
    experiment_output = cloudai.output.ExperimentOutput(experiment, tmp_path)
    first_run = cloudai.models.output.Run(
        path=str(tmp_path / "case" / "0"),
        jobid="101",
        status="completed",
        start=start,
        finish=start + datetime.timedelta(seconds=2),
        iteration=0,
        step=0,
    )
    second_run = cloudai.models.output.Run(
        path=str(tmp_path / "case" / "1"),
        jobid="102",
        status="running",
        start=start + datetime.timedelta(seconds=2),
        iteration=1,
        step=0,
    )

    experiment_output.update_run("case", first_run)
    experiment_output.update_run("case", second_run)
    second_run.status = "failed"
    second_run.finish = start + datetime.timedelta(seconds=4)
    experiment_output.update_run("case", second_run)
    experiment_output.finish("failed", start + datetime.timedelta(seconds=5))

    stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert stored.model_dump() == {
        "id": "experiment",
        "name": "scenario",
        "description": None,
        "status": "failed",
        "path": str(tmp_path),
        "start": start,
        "finish": start + datetime.timedelta(seconds=5),
        "duration": 5,
        "tests": [
            {
                "id": "case",
                "name": "workload",
                "description": None,
                "status": "failed",
                "path": str(tmp_path / "case"),
                "metrics": [],
                "runs": [
                    {
                        "path": str(tmp_path / "case" / "0"),
                        "jobid": "101",
                        "status": "completed",
                        "metrics": [],
                        "start": start,
                        "finish": start + datetime.timedelta(seconds=2),
                        "duration": 2,
                        "iteration": 0,
                        "step": 0,
                    },
                    {
                        "path": str(tmp_path / "case" / "1"),
                        "jobid": "102",
                        "status": "failed",
                        "metrics": [],
                        "start": start + datetime.timedelta(seconds=2),
                        "finish": start + datetime.timedelta(seconds=4),
                        "duration": 2,
                        "iteration": 1,
                        "step": 0,
                    },
                ],
                "dse": None,
            }
        ],
    }
