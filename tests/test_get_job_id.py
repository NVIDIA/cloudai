# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cloudai.core import JobIdRetrievalError, TestRun, TestScenario
from cloudai.systems.lsf.lsf_runner import LSFRunner
from cloudai.systems.lsf.lsf_system import LSFSystem
from cloudai.systems.slurm import SlurmJob, SlurmRunner, SlurmSystem
from cloudai.workloads.sleep.sleep import SleepCmdArgs, SleepTestDefinition


@pytest.fixture
def test_scenario(slurm_system: SlurmSystem) -> TestScenario:
    test_scenario = TestScenario(
        name="Test Scenario",
        test_runs=[
            TestRun(
                "tr-name",
                SleepTestDefinition(name="n", description="d", test_template_name="Sleep", cmd_args=SleepCmdArgs()),
                1,
                [],
                output_path=slurm_system.output_path / "tr-name",
            )
        ],
    )
    test_scenario.test_runs[0].output_path.mkdir(parents=True, exist_ok=True)
    return test_scenario


@pytest.fixture
def slurm_runner(slurm_system: SlurmSystem, test_scenario: TestScenario) -> SlurmRunner:
    return SlurmRunner(
        mode="run", system=slurm_system, test_scenario=test_scenario, output_path=slurm_system.output_path
    )


def test_job_id_retrieval_error(slurm_runner: SlurmRunner):
    tr = slurm_runner.test_scenario.test_runs[0]
    error = JobIdRetrievalError(
        test_name=str(tr.name),
        command="sbatch script.sh",
        stdout="",
        stderr="sbatch: error: Batch job submission failed: Requested node configuration is not available",
        message="Failed to retrieve job ID.",
    )
    with patch.object(SlurmSystem, "submit_job", side_effect=error), pytest.raises(JobIdRetrievalError) as excinfo:
        slurm_runner._submit_test(tr)
    assert "Failed to retrieve job ID." in str(excinfo.value)
    assert "sbatch: error: Batch job submission failed: Requested node configuration is not available" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "stdout, stderr, expected_job_id",
    [
        ("Submitted batch job 123456", "", 123456),
        ("submitted with Job ID 123456", "", 123456),
        ("", "sbatch: error: Batch job submission failed:...", None),
    ],
)
def test_slurm_get_job_id(stdout: str, stderr: str, expected_job_id: int | None):
    res = SlurmSystem._parse_submitted_job_id(stdout)
    assert res == expected_job_id


def test_slurm_runner_on_job_completion_calls_cleanup(slurm_runner: SlurmRunner):
    tr = slurm_runner.test_scenario.test_runs[0]
    job = SlurmJob(tr, id=1)
    slurm_runner.store_job_metadata = Mock()
    cleanup = Mock()
    slurm_runner.get_cmd_gen_strategy = Mock(return_value=Mock(cleanup_job_artifacts=cleanup))

    with patch.object(SlurmSystem, "complete_job", return_value=["node01", "node02"]) as complete_job:
        slurm_runner.on_job_completion(job)

    complete_job.assert_called_once_with(job)
    slurm_runner.store_job_metadata.assert_called_once_with(job)
    assert job.nodes == ["node01", "node02"]
    cleanup.assert_called_once()


def test_slurm_runner_records_and_reuses_nodes_per_case(slurm_runner: SlurmRunner, caplog: pytest.LogCaptureFixture):
    tr = slurm_runner.test_scenario.test_runs[0]
    tr.pin_nodes = True
    first_job = SlurmJob(tr, id=1)
    slurm_runner.store_job_metadata = Mock()
    slurm_runner.get_cmd_gen_strategy = Mock(return_value=Mock(cleanup_job_artifacts=Mock()))

    with patch.object(SlurmSystem, "complete_job", return_value=["node01", "node02"]):
        slurm_runner.on_job_completion(first_job)

    assert slurm_runner.pinned_nodes == {tr.name: ["node01", "node02"]}

    next_tr = TestRun(name=tr.name, test=tr.test, num_nodes=2, nodes=["bla1", "bla2"], pin_nodes=True)
    slurm_runner.on_job_submit = Mock()
    slurm_runner._submit_test = Mock(return_value=SlurmJob(next_tr, id=2))

    with caplog.at_level("INFO"):
        slurm_runner.submit_test(next_tr)

    assert next_tr.nodes == ["node01", "node02"]
    assert "Forcing test case 'tr-name' to use pinned nodes: node01,node02" in caplog.text
    slurm_runner.on_job_submit.assert_called_once_with(next_tr)


def test_slurm_runner_continues_when_pinned_case_has_no_recorded_nodes(
    slurm_runner: SlurmRunner, caplog: pytest.LogCaptureFixture
) -> None:
    tr = slurm_runner.test_scenario.test_runs[0]
    tr.pin_nodes = True
    job = SlurmJob(tr, id=1)
    slurm_runner.store_job_metadata = Mock()
    cleanup = Mock()
    slurm_runner.get_cmd_gen_strategy = Mock(return_value=Mock(cleanup_job_artifacts=cleanup))

    with patch.object(SlurmSystem, "complete_job", return_value=[]), caplog.at_level("ERROR"):
        slurm_runner.on_job_completion(job)

    assert tr.name not in slurm_runner.pinned_nodes
    assert "Cannot pin test case 'tr-name': the job has no recorded node allocation" in caplog.text
    slurm_runner.store_job_metadata.assert_called_once_with(job)
    cleanup.assert_called_once()


@pytest.mark.parametrize(
    "stdout, stderr, expected_job_id",
    [
        ("Job <123456> is submitted", "", 123456),
        ("", "error: ...", None),
    ],
)
def test_lsf_get_job_id(
    test_scenario: TestScenario, tmp_path: Path, stdout: str, stderr: str, expected_job_id: int | None
):
    lsf_runner = LSFRunner(
        mode="run",
        system=LSFSystem(name="test_system", install_path=Path(), output_path=tmp_path),
        test_scenario=test_scenario,
        output_path=tmp_path,
    )
    res = lsf_runner.get_job_id(stdout, stderr)
    assert res == expected_job_id
