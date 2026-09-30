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
    scenario.write_text(
        toml.dumps(
            {
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
        )
    )
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
    assert cloudai.api.list_experiments(system) == [(experiment.id, pathlib.Path(experiment.path))]
    assert signal.getsignal(signal.SIGINT) == signal_handler

    scenario.write_text('name = "invalid"\n')
    with pytest.raises(cloudai.core.TestScenarioParsingError):
        cloudai.api.run_experiment(scenario, system)
