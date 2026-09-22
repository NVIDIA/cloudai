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
import logging
import signal
from pathlib import Path
from unittest.mock import Mock

import pydantic
import pytest
import toml

import cloudai.api
import cloudai.core
import cloudai.handlers
import cloudai.metrics
from cloudai.systems.slurm import SingleSbatchRunner, SlurmRunner, SlurmSystem
from cloudai.systems.standalone import StandaloneSystem
from cloudai.systems.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone.standalone_runner import StandaloneRunner
from cloudai.workloads.sleep import SleepTestDefinition


@pytest.fixture
def configs(standalone_system: StandaloneSystem) -> tuple[str, str]:
    system = toml.dumps(
        {
            "name": standalone_system.name,
            "scheduler": "standalone",
            "install_path": str(standalone_system.install_path),
            "output_path": str(standalone_system.output_path),
            "monitor_interval": 0,
        }
    )
    scenario = toml.dumps(
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
    return scenario, system


@pytest.mark.parametrize("wait,successful", [(True, True), (True, False), (False, True)])
def test_experiment_api(
    configs: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait: bool, successful: bool
):
    scenario, system = configs
    process = Mock()
    monkeypatch.setattr("cloudai.handlers.subprocess.Popen", process)
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
        lambda *_: cloudai.core.JobStatusResult(successful, "test outcome"),
    )
    monkeypatch.setattr(
        SleepTestDefinition,
        "metric_observations",
        lambda *_: [cloudai.metrics.MetricObservation(cloudai.metrics.LATENCY, 2.0, {})],
    )
    handlers = logging.getLogger().handlers.copy()
    sigint = signal.getsignal(signal.SIGINT)
    assert cloudai.api.list_experiments(system) == []
    assert cloudai.api.validate_scenario(scenario, system) == (True, {})
    assert not (tmp_path / "output").exists()
    experiment = cloudai.api.run_experiment(scenario, system, wait=wait)
    if not wait:
        assert experiment.status == "pending"
        assert experiment.start is None
        assert cloudai.api.get_experiment(experiment.path) == experiment
        request_path = Path(experiment.path) / "request.json"
        assert process.call_args.args[0][-1] == str(request_path)
        cloudai.handlers.run_background(request_path)
        experiment = cloudai.api.get_experiment(experiment.path)
    else:
        process.assert_not_called()
    assert logging.getLogger().handlers == handlers
    assert signal.getsignal(signal.SIGINT) == sigint
    assert experiment.status == ("completed" if successful else "failed")
    assert experiment.start is not None and experiment.finish is not None
    test = experiment.tests[0]
    assert test.status == experiment.status
    assert len(test.runs) == 1
    assert test.runs[0].status == experiment.status
    expected_metrics = [{"name": "Latency", "value": 2.0, "unit": "us", "dimensions": []}] if successful else []
    assert [metric.model_dump() for metric in test.metrics] == expected_metrics
    assert test.metrics == test.runs[0].metrics
    result_dir = Path(experiment.path)
    assert cloudai.api.get_experiment(result_dir / "experiment.json") == experiment
    assert cloudai.api.list_experiments(system) == [(experiment.id, result_dir)]
    assert (result_dir / "system.toml").read_text() == system
    assert (result_dir / "scenario.toml").read_text() == scenario
    again = cloudai.api.run_experiment(scenario, system)
    assert again.id != experiment.id
    assert len(cloudai.api.list_experiments(system)) == 2
    (result_dir / "experiment.json").write_text("{}")
    with pytest.raises(pydantic.ValidationError):
        cloudai.api.get_experiment(result_dir)
    with pytest.raises(pydantic.ValidationError):
        cloudai.api.list_experiments(system)


def test_validate_scenario(configs: tuple[str, str], tmp_path: Path):
    scenario, system = configs
    system_path = tmp_path / "system.toml"
    system_path.write_text(system)
    template = toml.loads(scenario)["Tests"][0]
    template.pop("id")
    (tmp_path / "sleep.toml").write_text(toml.dumps(template))
    relative = 'name = "relative"\n[[Tests]]\nid = "sleep"\npath = "sleep.toml"\n'
    scenario_path = tmp_path / "scenario.toml"
    scenario_path.write_text(relative)
    assert cloudai.api.validate_scenario(scenario_path, system_path) == (True, {})
    valid, errors = cloudai.api.validate_scenario(relative, system)
    assert not valid and "Relative test paths" in errors["scenario"]
    for invalid in ("name = [", 'name = "invalid"'):
        valid, errors = cloudai.api.validate_scenario(invalid, system)
        assert not valid and errors["scenario"]
        with pytest.raises(cloudai.core.TestScenarioParsingError):
            cloudai.api.run_experiment(invalid, system)
    valid, errors = cloudai.api.validate_scenario(
        scenario, system.replace('scheduler = "standalone"', 'scheduler = "bad"')
    )
    assert not valid and "Unsupported system type" in errors["system"]
    valid, errors = cloudai.api.validate_scenario(scenario, system.replace('name = "standalone"', "name = 42"))
    assert not valid and "name" in errors["system"]
    valid, errors = cloudai.api.validate_scenario(scenario, tmp_path / "missing.toml")
    assert not valid and "missing.toml" in errors["system"]
    valid, errors = cloudai.api.validate_scenario(scenario, system, single_sbatch=True)
    assert not valid and "only supported for Slurm" in errors["scenario"]
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileNotFoundError):
        cloudai.api.get_experiment(tmp_path / "missing")


@pytest.mark.parametrize("failure", ["setup", "spawn", "worker-parse", "submission"])
def test_failed_experiment_is_saved(configs: tuple[str, str], monkeypatch: pytest.MonkeyPatch, failure: str):
    scenario, system = configs
    if failure == "setup":
        monkeypatch.setattr(StandaloneSystem, "update", Mock(side_effect=RuntimeError("setup failed")))
    elif failure == "spawn":
        monkeypatch.setattr("cloudai.handlers.subprocess.Popen", Mock(side_effect=OSError("spawn failed")))
    elif failure == "submission":
        monkeypatch.setattr(
            StandaloneRunner,
            "_submit_test",
            Mock(
                side_effect=cloudai.core.JobSubmissionError(
                    test_name="sleep", command="sleep 0", stdout="", stderr="", message="submission failed"
                )
            ),
        )
    if failure == "worker-parse":
        monkeypatch.setattr("cloudai.handlers.subprocess.Popen", Mock())
        pending = cloudai.api.run_experiment(scenario, system, wait=False)
        (Path(pending.path) / "scenario.toml").write_text("name = [")
        with pytest.raises(cloudai.core.TestScenarioParsingError):
            cloudai.handlers.run_background(Path(pending.path) / "request.json")
    else:
        with pytest.raises((RuntimeError, OSError, cloudai.core.JobSubmissionError)):
            cloudai.api.run_experiment(scenario, system, wait=failure != "spawn")
    experiments = cloudai.api.list_experiments(system)
    assert len(experiments) == 1
    failed = cloudai.api.get_experiment(experiments[0][1])
    assert failed.status == "failed"
    assert failed.finish is not None
    assert failed.tests[0].runs == []


def test_single_sbatch_selection_is_local(slurm_system: SlurmSystem, base_tr: cloudai.core.TestRun):
    scenario = cloudai.core.TestScenario(name="selection", test_runs=[base_tr])
    registered = cloudai.core.Registry().runners_map["slurm"]
    single = cloudai.handlers.create_experiment_runner(slurm_system, scenario, single_sbatch=True)
    normal = cloudai.handlers.create_experiment_runner(slurm_system, scenario)
    assert isinstance(single.runner, SingleSbatchRunner)
    assert type(normal.runner) is SlurmRunner
    assert cloudai.core.Registry().runners_map["slurm"] is registered
