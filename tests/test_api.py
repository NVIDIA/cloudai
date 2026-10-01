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
import cloudai.models.output
from cloudai.systems.standalone import StandaloneSystem
from cloudai.systems.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone.standalone_runner import StandaloneRunner
from cloudai.workloads.sleep import SleepTestDefinition


@pytest.fixture
def system_config(standalone_system: StandaloneSystem, tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "system.toml"
    path.write_text(
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
    return path


@pytest.fixture
def scenario_config(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scenario.toml"
    path.write_text(
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
    return path


@pytest.fixture
def mock_standalone_execution(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.fixture
def saved_experiment(
    standalone_system: StandaloneSystem,
) -> tuple[cloudai.models.output.Experiment, pathlib.Path]:
    experiment_path = standalone_system.output_path / "experiment-1"
    experiment_path.mkdir(parents=True)
    experiment = cloudai.models.output.Experiment(
        id=experiment_path.name,
        name="experiment",
        system_name=standalone_system.name,
        status="completed",
        path=str(experiment_path),
    )
    (experiment_path / "experiment.json").write_text(experiment.model_dump_json())
    return experiment, experiment_path


@pytest.mark.parametrize(("mode", "expected_runs"), [("run", 1), ("dry-run", 0)])
@pytest.mark.usefixtures("mock_standalone_execution")
def test_run_experiment(
    mode: str,
    expected_runs: int,
    system_config: pathlib.Path,
    scenario_config: pathlib.Path,
) -> None:
    signal_handler = signal.getsignal(signal.SIGINT)

    experiment = cloudai.api.run_experiment(scenario_config, system_config, mode=mode)

    assert experiment.status == "completed"
    assert len(experiment.tests[0].runs) == expected_runs
    assert experiment.id == pathlib.Path(experiment.path).name
    assert signal.getsignal(signal.SIGINT) == signal_handler


def test_run_experiment_rejects_single_sbatch_for_standalone(
    system_config: pathlib.Path,
    scenario_config: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="Single sbatch is only supported for Slurm systems"):
        cloudai.api.run_experiment(scenario_config, system_config, single_sbatch=True)


def test_run_experiment_rejects_invalid_scenario(
    system_config: pathlib.Path,
    scenario_config: pathlib.Path,
) -> None:
    scenario_config.write_text('name = "invalid"\n')

    with pytest.raises(cloudai.core.TestScenarioParsingError):
        cloudai.api.run_experiment(scenario_config, system_config)


def test_list_experiments(system_config: pathlib.Path, standalone_system: StandaloneSystem) -> None:
    assert cloudai.api.list_experiments(system_config) == []
    experiment_paths = [standalone_system.output_path / name for name in ("experiment-1", "experiment-2")]
    for path in experiment_paths:
        path.mkdir(parents=True)
        (path / "experiment.json").write_text("not parsed")
    (standalone_system.output_path / "ignored").mkdir()

    assert cloudai.api.list_experiments(system_config) == [(path.name, path.resolve()) for path in experiment_paths]


@pytest.mark.parametrize("reference", ["id", "path"])
def test_get_experiment(
    reference: str,
    system_config: pathlib.Path,
    saved_experiment: tuple[cloudai.models.output.Experiment, pathlib.Path],
) -> None:
    experiment, experiment_path = saved_experiment
    exp_id = experiment.id if reference == "id" else experiment_path

    assert cloudai.api.get_experiment(exp_id, system_config) == experiment


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("json-path", "Invalid experiment ID"),
        ("missing", "Invalid experiment ID"),
        ("traversal", "Invalid experiment ID"),
        ("malformed", "Cannot read experiment"),
    ],
)
def test_get_experiment_rejects_invalid_input(
    reference: str,
    message: str,
    system_config: pathlib.Path,
    saved_experiment: tuple[cloudai.models.output.Experiment, pathlib.Path],
) -> None:
    _, experiment_path = saved_experiment
    if reference == "json-path":
        exp_id: str | pathlib.Path = experiment_path / "experiment.json"
    elif reference == "missing":
        exp_id = "missing"
    elif reference == "traversal":
        exp_id = "../other"
    else:
        malformed_path = experiment_path.parent / "malformed"
        malformed_path.mkdir()
        (malformed_path / "experiment.json").write_text("not JSON")
        exp_id = malformed_path

    with pytest.raises(ValueError, match=message):
        cloudai.api.get_experiment(exp_id, system_config)
