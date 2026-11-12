# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import ConfigDict

from cloudai.core import (
    BaseJob,
    BaseRunner,
    JobFailureError,
    JobStatusResult,
    JobSubmissionError,
    System,
    TestDefinition,
    TestDependency,
    TestRun,
    TestScenario,
)
from cloudai.models.output import Experiment
from cloudai.models.workload import CmdArgs
from cloudai.systems.slurm import SlurmSystem


class MyRunner(BaseRunner):
    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path):
        super().__init__(mode, system, test_scenario, output_path)
        self.runner_job_status_result = JobStatusResult(is_successful=True)
        self.submitted_trs: list[TestRun] = []
        self.killed_by_dependency: list[BaseJob] = []

    def get_runner_job_status(self, job: BaseJob) -> JobStatusResult:
        return self.runner_job_status_result

    def _submit_test(self, tr: TestRun) -> BaseJob:
        self.submitted_trs.append(tr)
        return BaseJob(tr, 0)

    def delayed_submit_test(self, tr: TestRun, delay: int = 0):
        super().delayed_submit_test(tr, 0)

    def delayed_kill_job(self, job: BaseJob, delay: int = 0):
        self.killed_by_dependency.append(job)


class MyWorkload(TestDefinition):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workload_status_result: JobStatusResult = JobStatusResult(is_successful=True)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return self.workload_status_result


@pytest.fixture
def test_scenario(slurm_system: SlurmSystem) -> TestScenario:
    test = MyWorkload(name="n", description="d", test_template_name="wrk", cmd_args=CmdArgs())
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


@pytest.mark.parametrize("mode", ["run", "dry-run"])
@pytest.mark.parametrize("error", [OSError, KeyError, ImportError])
def test_username_lookup_failure_does_not_block_initialization(
    slurm_system: SlurmSystem,
    test_scenario: TestScenario,
    tmp_path: Path,
    mode: str,
    error: type[Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch("getpass.getuser", side_effect=error("Username unavailable")):
        runner = MyRunner(mode, slurm_system, test_scenario, tmp_path)
    runner.experiment_output.write()

    stored = Experiment.model_validate_json((tmp_path / "experiment.json").read_text())
    assert stored.user == ""
    assert stored.status == "running"
    assert stored.tests
    assert "Cannot determine the current username" in caplog.text


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
        tdef = cast(MyWorkload, tr.test)
        tdef.workload_status_result = JobStatusResult(is_successful=False, error_message="workload job failed")
        res = runner.get_job_status(job)
        assert not res.is_successful
        assert res.error_message == "workload job failed"

    def test_both_failed_runner_status_reported(self, runner: MyRunner):
        tr = runner.test_scenario.test_runs[0]
        job = BaseJob(tr, 0)
        tdef = cast(MyWorkload, tr.test)
        tdef.workload_status_result = JobStatusResult(is_successful=False, error_message="workload job failed")
        runner.runner_job_status_result = JobStatusResult(is_successful=False, error_message="runner job failed")
        res = runner.get_job_status(job)
        assert res == runner.runner_job_status_result


class TestHandleDependencies:
    """
    Tests for BaseRunner.handle_dependencies method.

    Both main and dependent TestRuns use the same MyWorkload test definition to reproduce an issue when jobs
    comparison was done incorrectly.
    """

    @pytest.fixture
    def tr_main(self, runner: MyRunner) -> TestRun:
        return runner.test_scenario.test_runs[0]

    def test_no_dependencies(self, runner: MyRunner, tr_main: TestRun):
        runner.handle_dependencies(BaseJob(tr_main, 0))
        assert len(runner.submitted_trs) == 0

    def test_start_post_comp(self, runner: MyRunner, tr_main: TestRun):
        tr_dep = deepcopy(tr_main)
        tr_dep.dependencies = {"start_post_comp": TestDependency(tr_main)}
        runner.test_scenario.test_runs.append(tr_dep)

        runner.handle_dependencies(BaseJob(tr_dep, 0))  # self, should not trigger anything
        assert len(runner.submitted_trs) == 0

        runner.handle_dependencies(BaseJob(tr_main, 0))
        assert len(runner.submitted_trs) == 1
        assert runner.submitted_trs[0] == tr_dep

    def test_end_post_comp(self, runner: MyRunner, tr_main: TestRun):
        tr_dep = deepcopy(tr_main)
        tr_dep.dependencies = {"end_post_comp": TestDependency(tr_main)}
        runner.test_scenario.test_runs.append(tr_dep)

        # self not running, main completed -> nothing to kill
        runner.handle_dependencies(BaseJob(tr_main, 0))
        assert len(runner.killed_by_dependency) == 0

        # self is running, main completed -> should kill
        runner.submit_test(tr_dep)

        runner.handle_dependencies(BaseJob(tr_dep, 0))  # self, should not kill
        assert len(runner.killed_by_dependency) == 0

        runner.handle_dependencies(BaseJob(tr_main, 0))
        assert len(runner.killed_by_dependency) == 1
        assert runner.killed_by_dependency[0].test_run == tr_dep


class TestPollJobs:
    @pytest.fixture
    def tr_main(self, runner: MyRunner) -> TestRun:
        return runner.test_scenario.test_runs[0]

    @pytest.fixture
    def run_mode(self, slurm_system: SlurmSystem, test_scenario: TestScenario) -> MyRunner:
        return MyRunner("run", slurm_system, test_scenario, slurm_system.output_path)

    def test_asks_for_completion_once_and_never_for_running_state(self, run_mode: MyRunner, tr_main: TestRun):
        run_mode.submit_test(tr_main)

        with (
            patch.object(SlurmSystem, "is_job_completed", side_effect=[False, False, True]) as is_completed,
            patch.object(SlurmSystem, "is_job_running") as is_running,
        ):
            assert run_mode.poll_jobs() == []
            assert run_mode.poll_jobs() == []
            [(job, status)] = run_mode.poll_jobs()

        assert is_completed.call_count == 3
        is_running.assert_not_called()
        assert job.test_run is tr_main
        assert status.is_successful
        assert run_mode.jobs == []
        assert run_mode.testrun_to_job_map == {}

    def test_failed_job_is_only_reported(self, run_mode: MyRunner, tr_main: TestRun):
        cast(MyWorkload, tr_main.test).workload_status_result = JobStatusResult(False, "workload job failed")
        tr_other = deepcopy(tr_main)
        tr_other.name = "other"
        run_mode.submit_test(tr_main)
        run_mode.submit_test(tr_other)

        with patch.object(SlurmSystem, "is_job_completed", side_effect=[True, False]):
            [(job, status)] = run_mode.poll_jobs()

        assert job.test_run is tr_main
        assert (status.is_successful, status.error_message) == (False, "workload job failed")
        assert [job.test_run for job in run_mode.jobs] == [tr_other]
        assert not run_mode.shutting_down

    def test_start_post_init_dependent_is_submitted_once_the_job_runs(self, run_mode: MyRunner, tr_main: TestRun):
        tr_dep = deepcopy(tr_main)
        tr_dep.name = "dependent"
        tr_dep.dependencies = {"start_post_init": TestDependency(tr_main)}
        run_mode.test_scenario.test_runs.append(tr_dep)
        run_mode.submit_test(tr_main)

        with (
            patch.object(SlurmSystem, "is_job_completed", return_value=False),
            patch.object(SlurmSystem, "is_job_running", side_effect=[False, True]) as is_running,
        ):
            run_mode.poll_jobs()
            assert run_mode.submitted_trs == [tr_main]
            run_mode.poll_jobs()
            assert run_mode.submitted_trs == [tr_main, tr_dep]
            run_mode.poll_jobs()

        assert is_running.call_count == 2

    def test_dry_run_completes_without_asking_the_system(self, runner: MyRunner, tr_main: TestRun):
        runner.submit_test(tr_main)

        with patch.object(SlurmSystem, "is_job_completed") as is_completed:
            [(job, status)] = runner.poll_jobs()

        is_completed.assert_not_called()
        assert job.test_run is tr_main
        assert status.is_successful


class TestMonitorJobs:
    @pytest.fixture
    def tr_main(self, runner: MyRunner) -> TestRun:
        return runner.test_scenario.test_runs[0]

    @pytest.fixture
    def run_mode(self, slurm_system: SlurmSystem, test_scenario: TestScenario) -> MyRunner:
        return MyRunner("run", slurm_system, test_scenario, slurm_system.output_path)

    def test_failed_job_stops_the_scenario(self, run_mode: MyRunner, tr_main: TestRun):
        cast(MyWorkload, tr_main.test).workload_status_result = JobStatusResult(False, "workload job failed")
        tr_other = deepcopy(tr_main)
        tr_other.name = "other"
        run_mode.submit_test(tr_main)
        run_mode.submit_test(tr_other)

        with (
            patch.object(SlurmSystem, "is_job_completed", side_effect=[True, False]),
            patch.object(SlurmSystem, "kill") as kill,
            pytest.raises(JobFailureError),
        ):
            run_mode.monitor_jobs()

        assert run_mode.shutting_down
        assert [call.args[0].test_run for call in kill.call_args_list] == [tr_other]

    def test_failed_job_is_tolerated_without_job_status_check(self, run_mode: MyRunner, tr_main: TestRun):
        cast(MyWorkload, tr_main.test).workload_status_result = JobStatusResult(False, "workload job failed")
        run_mode.test_scenario.job_status_check = False
        run_mode.submit_test(tr_main)

        with patch.object(SlurmSystem, "is_job_completed", return_value=True):
            assert run_mode.monitor_jobs() == 1

        assert not run_mode.shutting_down
        assert run_mode.jobs == []

    def test_job_is_resubmitted_for_every_iteration(self, runner: MyRunner, tr_main: TestRun):
        tr_main.iterations = 3
        runner.submit_test(tr_main)

        assert [runner.monitor_jobs() for _ in range(3)] == [1, 1, 1]

        assert tr_main.current_iteration == 2
        assert runner.submitted_trs == [tr_main] * 3
        assert runner.jobs == []


class TestRunScenario:
    @pytest.fixture
    def tr_main(self, runner: MyRunner) -> TestRun:
        return runner.test_scenario.test_runs[0]

    def test_submission_error_kills_outstanding_jobs(self, runner: MyRunner, tr_main: TestRun):
        tr_broken = deepcopy(tr_main)
        tr_broken.name = "broken"
        runner.test_scenario.test_runs.append(tr_broken)
        error = JobSubmissionError("broken", "cmd", "", "", "cannot submit")
        submit = runner._submit_test

        def submit_or_fail(tr: TestRun) -> BaseJob:
            if tr is tr_broken:
                raise error
            return submit(tr)

        with (
            patch.object(runner, "_submit_test", side_effect=submit_or_fail),
            patch.object(SlurmSystem, "kill") as kill,
            pytest.raises(JobSubmissionError),
        ):
            runner.run()

        assert runner.shutting_down
        assert [call.args[0].test_run for call in kill.call_args_list] == [tr_main]
