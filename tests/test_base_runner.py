# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, final
from unittest.mock import Mock

import pytest

from cloudai import BaseJob, System, Test, TestRun, TestScenario
from cloudai._core.base_runner import NewBaseRunner
from cloudai._core.cases_iter import CasesIter, StaticCasesListIter
from cloudai._core.job_status_result import JobStatusResult
from cloudai._core.test_scenario import TestDependency
from cloudai._core.test_template import TestTemplate
from cloudai.test_definitions.sleep import SleepCmdArgs, SleepTestDefinition


class MySystem(System):
    def __init__(
        self,
        name: str,
        scheduler: str,
        install_path: Path,
        output_path: Path,
        global_env_vars: Optional[Dict[str, Any]] = None,
        monitor_interval: int = 1,
    ) -> None:
        super().__init__(name, scheduler, install_path, output_path, global_env_vars, monitor_interval)
        self._job_id_counter: int = 0

    @property
    def job_id_counter(self) -> int:
        curr = self._job_id_counter
        self._job_id_counter += 1
        return curr

    def update(self) -> None:
        return

    def is_job_running(self, job: BaseJob) -> bool:
        return False

    def is_job_completed(self, job: BaseJob) -> bool:
        return True

    def kill(self, job: BaseJob) -> None:
        return


class SlurmSbatchRunner(NewBaseRunner):
    """Runner that submits jobs to Slurm using sbatch.

    This class gets all available tasks from scenario iterator and submits it into the system. Jobs are monitored and
    when new jobs are available, they are submitted to the system too. Each job is an sbatch run.
    """

    def __init__(self, mode: str, system: MySystem, test_scenario_iter: CasesIter):
        self.mode = mode
        self.system = system
        self.test_scenario_iter = test_scenario_iter

        self.active_jobs: dict[str, BaseJob] = {}
        self.completed_jobs: dict[str, BaseJob] = {}

    @final
    def run(self):
        while self.test_scenario_iter.has_more_cases:
            for tr in self.test_scenario_iter:
                self.submit_one(tr)

            self.process_completed_jobs()
            self.clean_active_jobs()

            time.sleep(self.system.monitor_interval)

        while self.active_jobs:
            self.process_completed_jobs()
            self.clean_active_jobs()
            time.sleep(self.system.monitor_interval)

    def process_completed_jobs(self):
        for job in self.active_jobs.values():
            if self.system.is_job_completed(job):
                self.completed_jobs[job.test_run.name] = job
                self.test_scenario_iter.on_completed(job.test_run, self)

    def clean_active_jobs(self):
        in_common = set(self.active_jobs.keys()) & set(self.completed_jobs.keys())
        for name in in_common:
            del self.active_jobs[name]

    def submit_one(self, tr: TestRun) -> None:
        job = BaseJob(tr, self.system.job_id_counter)
        self.active_jobs[tr.name] = job

    def kill_one(self, tr: TestRun) -> None:
        job = self.active_jobs.get(tr.name)
        if job:
            self.system.kill(job)


@pytest.fixture
def system(tmp_path: Path) -> MySystem:
    return MySystem(
        "system",
        "scheduler",
        tmp_path / "install",
        tmp_path / "output",
        monitor_interval=0.01,  # type: ignore
    )


@pytest.fixture
def test(system: MySystem) -> Test:
    tdef = SleepTestDefinition(name="test", description="desc", test_template_name="template", cmd_args=SleepCmdArgs())
    ttempl = TestTemplate(name="template", system=system)
    ttempl.get_job_id = Mock(return_value=0)
    ttempl.get_job_status = Mock(return_value=JobStatusResult(True))
    test = Test(test_definition=tdef, test_template=ttempl)
    return test


@pytest.fixture
def partial_tr(system: MySystem, test: Test) -> partial[TestRun]:
    return partial(TestRun, test=test, num_nodes=1, nodes=[], output_path=system.output_path)


class TestStaticScenarioIter:
    def test_one_run(self, partial_tr: partial[TestRun]):
        tr = partial_tr(name="tr")

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[tr]))
        cases = [tr for tr in ssi]

        assert len(cases) == 1
        assert cases[0].name == tr.name

    def test_two_independent_runs(self, partial_tr: partial[TestRun]):
        tr1, tr2 = partial_tr(name="tr1"), partial_tr(name="tr2")

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[tr1, tr2]))
        cases = [tr for tr in ssi]

        assert len(cases) == 2
        assert cases[0].name == tr1.name
        assert cases[1].name == tr2.name

    def test_depends_on_init(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"start_post_init": TestDependency(main_tr)})

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[dep_tr, main_tr]))

        cases = [tr for tr in ssi]
        assert len(cases) == 2
        assert cases[0].name == main_tr.name
        assert cases[1].name == dep_tr.name
        assert not ssi.has_more_cases

    def test_depends_on_comp(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"start_post_comp": TestDependency(main_tr)})

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[dep_tr, main_tr]))

        # cycle one, only independent runs
        cases = [tr for tr in ssi]
        assert len(cases) == 1
        assert cases[0].name == main_tr.name
        assert ssi.has_more_cases

        # cycle two, dependent run
        cases = [tr for tr in ssi]
        assert len(cases) == 0
        assert ssi.has_more_cases

        # cycle three, mark as competed to allow dependent run
        ssi.on_completed(main_tr, Mock())
        cases = [tr for tr in ssi]
        assert len(cases) == 1
        assert not ssi.has_more_cases

    def test_to_kill_on_comp(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"end_post_comp": TestDependency(main_tr)})

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[dep_tr, main_tr]))

        # cycle one, both runs
        cases = [tr for tr in ssi]
        assert len(cases) == 2
        assert not ssi.has_more_cases

        runner = Mock()
        ssi.on_completed(main_tr, runner)
        assert runner.kill.called_once_with(dep_tr)


class TestMyRunner:
    def test_two_independent_runs(self, partial_tr: partial[TestRun], system: MySystem):
        tr1, tr2 = partial_tr(name="tr1"), partial_tr(name="tr2")

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[tr1, tr2]))
        runner = SlurmSbatchRunner("run", system, ssi)
        runner.run()

        assert len(runner.active_jobs) == 0
        assert len(runner.completed_jobs) == 2
        assert runner.completed_jobs[tr1.name].id == 0
        assert runner.completed_jobs[tr1.name].test_run.name == tr1.name
        assert runner.completed_jobs[tr2.name].id == 1
        assert runner.completed_jobs[tr2.name].test_run.name == tr2.name

    def test_two_dependent_runs(self, partial_tr: partial[TestRun], system: MySystem):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"start_post_comp": TestDependency(main_tr)})

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[dep_tr, main_tr]))
        runner = SlurmSbatchRunner("run", system, ssi)
        runner.run()

        assert len(runner.active_jobs) == 0
        assert len(runner.completed_jobs) == 2
        assert runner.completed_jobs[main_tr.name].id == 0
        assert runner.completed_jobs[main_tr.name].test_run.name == main_tr.name
        assert runner.completed_jobs[dep_tr.name].id == 1
        assert runner.completed_jobs[dep_tr.name].test_run.name == dep_tr.name

    def test_two_dependencies(self, partial_tr: partial[TestRun], system: MySystem):
        main_tr = partial_tr(name="tr-main")
        post_comp_tr = partial_tr(name="tr-post_comp", dependencies={"start_post_comp": TestDependency(main_tr)})
        post_init_tr = partial_tr(name="tr-post_init", dependencies={"start_post_init": TestDependency(main_tr)})

        ssi = StaticCasesListIter(TestScenario(name="scenario", test_runs=[post_comp_tr, post_init_tr, main_tr]))
        runner = SlurmSbatchRunner("run", system, ssi)
        runner.run()

        assert len(runner.active_jobs) == 0
        assert len(runner.completed_jobs) == 3
        assert runner.completed_jobs[main_tr.name].id == 0
        assert runner.completed_jobs[main_tr.name].test_run.name == main_tr.name
        assert runner.completed_jobs[post_init_tr.name].id == 1
        assert runner.completed_jobs[post_init_tr.name].test_run.name == post_init_tr.name
        assert runner.completed_jobs[post_comp_tr.name].id == 2
        assert runner.completed_jobs[post_comp_tr.name].test_run.name == post_comp_tr.name
