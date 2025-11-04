# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import cast

import pytest
from pydantic import ConfigDict

from cloudai._core.system import System
from cloudai.core import BaseJob, BaseRunner, JobStatusResult, Test, TestDefinition, TestRun, TestScenario
from cloudai.models.workload import CmdArgs
from cloudai.systems.slurm import SlurmSystem


class MyRunner(BaseRunner):
    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path):
        super().__init__(mode, system, test_scenario, output_path)
        self.runner_job_status_result = JobStatusResult(is_successful=True)

    def get_runner_job_status(self, job: BaseJob) -> JobStatusResult:
        return self.runner_job_status_result

    def _submit_test(self, tr: TestRun) -> BaseJob:
        return BaseJob(tr, 0)


class MyWorkload(TestDefinition):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workload_status_result: JobStatusResult = JobStatusResult(is_successful=True)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return self.workload_status_result


@pytest.fixture
def test_scenario(slurm_system: SlurmSystem) -> TestScenario:
    test = Test(test_definition=MyWorkload(name="n", description="d", test_template_name="wrk", cmd_args=CmdArgs()))
    test_scenario = TestScenario(
        name="Test Scenario",
        test_runs=[TestRun("tr-name", test, 1, [], output_path=slurm_system.output_path / "tr-name")],
    )
    return test_scenario


@pytest.fixture
def runner(slurm_system: SlurmSystem, test_scenario: TestScenario) -> MyRunner:
    return MyRunner(
        mode="dry-run", system=slurm_system, test_scenario=test_scenario, output_path=slurm_system.output_path
    )


class TestGetJobStatus:
    def test_both_successful(self, runner: MyRunner):
        job = BaseJob(runner.test_scenario.test_runs[0], 0)
        assert runner.get_job_status(job).is_successful

    def test_runner_job_failed(self, runner: MyRunner):
        job = BaseJob(runner.test_scenario.test_runs[0], 0)
        runner.runner_job_status_result = JobStatusResult(is_successful=False, error_message="runner job failed")
        res = runner.get_job_status(job)
        assert not res.is_successful
        assert res.error_message == "runner job failed"

    def test_workload_job_failed(self, runner: MyRunner):
        tr = runner.test_scenario.test_runs[0]
        job = BaseJob(tr, 0)
        tdef = cast(MyWorkload, tr.test.test_definition)
        tdef.workload_status_result = JobStatusResult(is_successful=False, error_message="workload job failed")
        res = runner.get_job_status(job)
        assert not res.is_successful
        assert res.error_message == "workload job failed"

    def test_both_failed_runner_status_reported(self, runner: MyRunner):
        tr = runner.test_scenario.test_runs[0]
        job = BaseJob(tr, 0)
        tdef = cast(MyWorkload, tr.test.test_definition)
        tdef.workload_status_result = JobStatusResult(is_successful=False, error_message="workload job failed")
        runner.runner_job_status_result = JobStatusResult(is_successful=False, error_message="runner job failed")
        res = runner.get_job_status(job)
        assert res == runner.runner_job_status_result
