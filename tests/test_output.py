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

from cloudai.models import output as output_models
from cloudai.output import ExperimentOutput


def test_experiment_output_preserves_runs_and_finalizes_failure(tmp_path: pathlib.Path) -> None:
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    experiment = output_models.Experiment(
        id="experiment",
        name="scenario",
        status="running",
        path=str(tmp_path),
        start=start,
        tests=[output_models.Test(id="case", name="workload", path=str(tmp_path / "case"))],
    )
    experiment_output = ExperimentOutput(experiment, tmp_path)
    first_run = output_models.Run(
        path=str(tmp_path / "case" / "0"),
        jobid="101",
        status="completed",
        start=start,
        finish=start + datetime.timedelta(seconds=2),
        iteration=0,
        step=0,
    )
    second_run = output_models.Run(
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

    stored = output_models.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert stored.status == "failed"
    assert stored.duration == 5
    assert stored.tests[0].status == "failed"
    assert [(run.jobid, run.status, run.duration) for run in stored.tests[0].runs] == [
        ("101", "completed", 2),
        ("102", "failed", 2),
    ]
