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
import signal

import pytest
import toml

import cloudai.api
import cloudai.core
from cloudai.systems.standalone import StandaloneSystem
from cloudai.systems.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone.standalone_runner import StandaloneRunner
from cloudai.workloads.sleep import SleepTestDefinition


def test_run_and_list_experiments(
    standalone_system: StandaloneSystem, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system.toml"
    system.write_text(
        toml.dumps(
            {
                "name": standalone_system.name,
                "scheduler": "standalone",
                "install_path": str(standalone_system.install_path),
                "output_path": str(standalone_system.output_path),
                "monitor_interval": 0,
            }
        )
    )
    scenario = tmp_path / "scenario.toml"
    scenario_data = {
        "name": "api-test",
        "Tests": [
            {
                "id": "sleep",
                "name": "sleep",
                "description": "API smoke test",
                "test_template_name": "Sleep",
                "cmd_args": {"seconds": 0},
            }
        ],
    }
    scenario.write_text(toml.dumps(scenario_data))
    monkeypatch.setattr(StandaloneSystem, "is_job_completed", lambda *_: True)
    monkeypatch.setattr(StandaloneSystem, "is_job_running", lambda *_: False)
    monkeypatch.setattr(
        StandaloneRunner,
        "_submit_test",
        lambda self, tr: StandaloneJob(tr, id=123, start=datetime.datetime.now(datetime.timezone.utc)),
    )
    monkeypatch.setattr(
        SleepTestDefinition,
        "was_run_successful",
        lambda *_: cloudai.core.JobStatusResult(True, "test outcome"),
    )

    signal_handler = signal.getsignal(signal.SIGINT)
    assert cloudai.api.list_experiments(system) == []
    experiment = cloudai.api.run_experiment(scenario, system)
    assert experiment.status == "completed"
    assert experiment.tests[0].runs[0].status == "completed"
    experiment_path = pathlib.Path(experiment.path)
    assert experiment.id == experiment_path.name
    unparsed_path = standalone_system.output_path / "zzz-unparsed"
    unparsed_path.mkdir()
    (unparsed_path / "experiment.json").write_text("not JSON")
    assert cloudai.api.list_experiments(system) == [
        (experiment.id, experiment_path),
        (unparsed_path.name, unparsed_path.resolve()),
    ]
    assert cloudai.api.get_experiment(experiment.id, str(system)) == experiment
    assert cloudai.api.get_experiment(pathlib.Path(experiment.path), system) == experiment
    with pytest.raises(ValueError, match="Invalid experiment ID"):
        cloudai.api.get_experiment(pathlib.Path(experiment.path) / "experiment.json", system)
    with pytest.raises(ValueError, match="Invalid experiment ID"):
        cloudai.api.get_experiment("missing", system)
    with pytest.raises(ValueError, match="Invalid experiment ID"):
        cloudai.api.get_experiment("../other", system)
    with pytest.raises(ValueError, match="Cannot read experiment"):
        cloudai.api.get_experiment(unparsed_path, system)
    assert signal.getsignal(signal.SIGINT) == signal_handler

    scenario_data["name"] = "api-dry-run"
    scenario.write_text(toml.dumps(scenario_data))
    dry_run_experiment = cloudai.api.run_experiment(scenario, system, mode="dry-run")
    assert dry_run_experiment.tests[0].runs == []
    with pytest.raises(ValueError, match="Single sbatch is only supported for Slurm systems"):
        cloudai.api.run_experiment(scenario, system, single_sbatch=True)

    scenario.write_text('name = "invalid"\n')
    with pytest.raises(cloudai.core.TestScenarioParsingError):
        cloudai.api.run_experiment(scenario, system)
