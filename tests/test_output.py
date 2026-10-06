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

import copy
import datetime
import pathlib

import pytest

import cloudai.core
import cloudai.metrics
import cloudai.models.output
import cloudai.output


def test_refresh_metrics_preserves_dse_selection_and_failed_runs(
    tmp_path: pathlib.Path,
    base_tr: cloudai.core.TestRun,
    slurm_system: cloudai.core.System,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def observations(self, system, tr):
        return [cloudai.metrics.MetricObservation(cloudai.metrics.ITERATION_TIME, float(tr.step), {})]

    monkeypatch.setattr(cloudai.core.TestDefinition, "metric_observations", observations)
    selected = cloudai.models.output.DSE(space={"size": [1, 2]}, best_step=2, best_config={"size": 2})
    experiment = cloudai.models.output.Experiment(
        id="experiment",
        name="scenario",
        system_name="remote",
        path="/remote/results",
        status="completed",
        tests=[
            cloudai.models.output.Test(
                id=str(base_tr.name),
                name="workload",
                path="/remote/results/case",
                status="failed",
                dse=selected,
                runs=[
                    cloudai.models.output.Run(
                        path=f"/remote/results/case/0/{step}",
                        jobid=str(step),
                        status="failed" if step == 3 else "completed",
                        iteration=0,
                        step=step,
                    )
                    for step in (1, 2, 3)
                ],
            )
        ],
    )
    (tmp_path / "experiment.json").write_text(experiment.model_dump_json())
    test_runs = []
    for step in (1, 2, 3):
        tr = copy.deepcopy(base_tr)
        tr.step = step
        tr.output_path = tmp_path / "case" / "0" / str(step)
        test_runs.append(tr)
    cloudai.output.refresh_experiment_metrics(slurm_system, test_runs, tmp_path)
    refreshed = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    test = refreshed.tests[0]
    assert test.dse == selected
    assert [run.metrics[0].value for run in test.runs[:2]] == [1.0, 2.0]
    assert test.metrics == test.runs[1].metrics
    assert test.runs[2] == experiment.tests[0].runs[2]
    assert test.status == "failed"
    assert refreshed.path == "/remote/results"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_experiment_output_preserves_runs_and_finalizes_failure(
    tmp_path: pathlib.Path, status: cloudai.models.output.Status
) -> None:
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    experiment = cloudai.models.output.Experiment(
        id="experiment",
        name="scenario",
        system_name="test-system",
        status="running",
        path=str(tmp_path),
        start=start,
        tests=[
            cloudai.models.output.Test(id=case, name="workload", path=str(tmp_path / case))
            for case in ("case", "interrupted", "not-started")
        ],
    )
    experiment_output = cloudai.output.ExperimentOutput(experiment, tmp_path)
    first_run = cloudai.models.output.Run(
        path=str(tmp_path / "case" / "0"),
        jobid="101",
        status="completed",
        metrics=[cloudai.models.output.Metric(name="Bandwidth", value=12.5, unit="GB/s")],
        start=start,
        finish=start + datetime.timedelta(seconds=2),
        duration=2,
        iteration=0,
        step=1,
    )
    second_run = cloudai.models.output.Run(
        path=str(tmp_path / "case" / "1"),
        jobid="102",
        status="pending",
        start=start + datetime.timedelta(seconds=2),
        iteration=1,
        step=2,
    )

    experiment_output.update_run("case", first_run)
    experiment_output.update_run("case", second_run)
    assert experiment_output.snapshot().tests[0].status == "pending"
    second_run.status = "failed"
    second_run.finish = start + datetime.timedelta(seconds=4)
    second_run.duration = 2
    experiment_output.update_run("case", second_run)
    experiment_output.update_dse(
        "case",
        {"algorithm": ["first", "second"]},
        [(2, {"algorithm": "second"}), (1, {"algorithm": "first"})],
    )
    experiment_output.update_run(
        "interrupted",
        cloudai.models.output.Run(path=str(tmp_path / "interrupted" / "0"), jobid="103", status="pending"),
    )
    experiment_output.finish(status, start + datetime.timedelta(seconds=5))

    stored = cloudai.models.output.Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert stored.model_dump() == {
        "id": "experiment",
        "name": "scenario",
        "system_name": "test-system",
        "description": None,
        "status": "cancelled" if status == "cancelled" else "failed",
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
                "metrics": [{"name": "Bandwidth", "value": 12.5, "unit": "GB/s", "dimensions": []}],
                "runs": [
                    {
                        "path": str(tmp_path / "case" / "0"),
                        "jobid": "101",
                        "status": "completed",
                        "metrics": [{"name": "Bandwidth", "value": 12.5, "unit": "GB/s", "dimensions": []}],
                        "start": start,
                        "finish": start + datetime.timedelta(seconds=2),
                        "duration": 2,
                        "iteration": 0,
                        "step": 1,
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
                        "step": 2,
                    },
                ],
                "dse": {
                    "space": {"algorithm": ["first", "second"]},
                    "best_config": {"algorithm": "first"},
                    "best_step": 1,
                },
            },
            {
                "id": "interrupted",
                "name": "workload",
                "description": None,
                "status": "unknown",
                "path": str(tmp_path / "interrupted"),
                "metrics": [],
                "runs": [
                    {
                        "path": str(tmp_path / "interrupted" / "0"),
                        "jobid": "103",
                        "status": "unknown",
                        "metrics": [],
                        "start": None,
                        "finish": None,
                        "duration": None,
                        "iteration": None,
                        "step": None,
                    }
                ],
                "dse": None,
            },
            {
                "id": "not-started",
                "name": "workload",
                "description": None,
                "status": "unknown",
                "path": str(tmp_path / "not-started"),
                "metrics": [],
                "runs": [],
                "dse": None,
            },
        ],
    }
