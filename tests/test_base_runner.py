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
from abc import ABC, abstractmethod
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, final
from unittest.mock import Mock

import pytest
from cloudai import BaseJob, System, Test, TestRun, TestScenario
from cloudai._core.job_status_result import JobStatusResult
from cloudai._core.test_scenario import TestDependency
from cloudai._core.test_template import TestTemplate
from cloudai.test_definitions.sleep import SleepCmdArgs, SleepTestDefinition


def create_autospec_dataclass(dataclass: type) -> TestRun:
    return Mock(spec=[field.name for field in fields(dataclass)])


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


class Runner(ABC):
    @abstractmethod
    def submit_one(self, tr: TestRun) -> None: ...

    @abstractmethod
    def kill_one(self, tr: TestRun) -> None: ...


class StaticScenario(TestScenario):
    def __init__(self, name: str, test_runs: list[TestRun]):
        super().__init__(name, test_runs)
        self.ready_for_run: list[TestRun] = []
        self.submitted: set[str] = set()
        self.completed: set[str] = set()
        self.stop_on_completion: dict[str, list[TestRun]] = {}

    @property
    def has_more_runs(self) -> bool:
        return len(self.submitted) < len(self.test_runs)

    def __iter__(self):
        not_submitted = [tr for tr in self.test_runs if tr.name not in self.submitted]
        for tr in not_submitted:
            if not tr.dependencies:
                self.ready_for_run.append(tr)
            elif tr.dependencies:
                if "start_post_comp" in tr.dependencies:
                    dep = tr.dependencies["start_post_comp"]
                    if dep.test_run.name in self.completed:
                        self.ready_for_run.append(tr)
                elif "end_post_comp" in tr.dependencies:
                    self.ready_for_run.append(tr)
                    dep = tr.dependencies["end_post_comp"]
                    self.stop_on_completion.setdefault(dep.test_run.name, []).append(tr)

        for tr in not_submitted:  # submit post_init dependencies right after main
            if tr.dependencies and "start_post_init" in tr.dependencies:
                dep = tr.dependencies["start_post_init"]
                if dep.test_run.name in {t.name for t in self.ready_for_run}:
                    self.ready_for_run.append(tr)

        return self

    def __next__(self) -> TestRun:
        if not self.ready_for_run:
            raise StopIteration

        self.submitted.add(self.ready_for_run[0].name)
        return self.ready_for_run.pop(0)

    def on_completed(self, tr: TestRun, runner: Runner) -> None:
        self.completed.add(tr.name)
        for tr_to_stop in self.stop_on_completion.get(tr.name, []):
            runner.kill_one(tr_to_stop)


class MyRunner(Runner):
    def __init__(self, mode: str, system: MySystem, test_scenario: StaticScenario):
        self.mode = mode
        self.system = system
        self.test_scenario = test_scenario

        self.active_jobs: dict[str, BaseJob] = {}
        self.completed_jobs: dict[str, BaseJob] = {}

    @final
    def run(self):
        while self.test_scenario.has_more_runs:
            for tr in self.test_scenario:
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
            if job.is_completed():
                self.completed_jobs[job.test_run.name] = job
                self.test_scenario.on_completed(job.test_run, self)

    def clean_active_jobs(self):
        in_common = set(self.active_jobs.keys()) & set(self.completed_jobs.keys())
        for name in in_common:
            del self.active_jobs[name]

    def submit_one(self, tr: TestRun) -> None:
        job = BaseJob(self.mode, self.system, tr)
        job.id = self.system.job_id_counter
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


class TestStaticScenario:
    def test_one_run(self, partial_tr: partial[TestRun]):
        tr = partial_tr(name="tr")

        ts = StaticScenario(name="scenario", test_runs=[tr])
        cases = [tr for tr in ts]

        assert len(cases) == 1
        assert cases[0].name == tr.name

    def test_two_independent_runs(self, partial_tr: partial[TestRun]):
        tr1, tr2 = partial_tr(name="tr1"), partial_tr(name="tr2")

        ts = StaticScenario(name="scenario", test_runs=[tr1, tr2])
        cases = [tr for tr in ts]

        assert len(cases) == 2
        assert cases[0].name == tr1.name
        assert cases[1].name == tr2.name

    def test_depends_on_init(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"start_post_init": TestDependency(main_tr)})

        ts = StaticScenario(name="scenario", test_runs=[dep_tr, main_tr])

        cases = [tr for tr in ts]
        assert len(cases) == 2
        assert cases[0].name == main_tr.name
        assert cases[1].name == dep_tr.name
        assert not ts.has_more_runs

    def test_depends_on_comp(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"start_post_comp": TestDependency(main_tr)})

        ts = StaticScenario(name="scenario", test_runs=[dep_tr, main_tr])

        # cycle one, only independent runs
        cases = [tr for tr in ts]
        assert len(cases) == 1
        assert cases[0].name == main_tr.name
        assert ts.has_more_runs

        # cycle two, dependent run
        cases = [tr for tr in ts]
        assert len(cases) == 0
        assert ts.has_more_runs

        # cycle three, mark as competed to allow dependent run
        ts.on_completed(main_tr, Mock())
        cases = [tr for tr in ts]
        assert len(cases) == 1
        assert not ts.has_more_runs

    def test_to_kill_on_comp(self, partial_tr: partial[TestRun]):
        main_tr = partial_tr(name="tr-main")
        dep_tr = partial_tr(name="tr-dep", dependencies={"end_post_comp": TestDependency(main_tr)})

        ts = StaticScenario(name="scenario", test_runs=[dep_tr, main_tr])

        # cycle one, both runs
        cases = [tr for tr in ts]
        assert len(cases) == 2
        assert not ts.has_more_runs

        runner = Mock()
        ts.on_completed(main_tr, runner)
        assert runner.kill.called_once_with(dep_tr)


class TestMyRunner:
    def test_two_independent_runs(self, partial_tr: partial[TestRun], system: MySystem):
        tr1, tr2 = partial_tr(name="tr1"), partial_tr(name="tr2")

        ts = StaticScenario(name="scenario", test_runs=[tr1, tr2])
        runner = MyRunner("run", system, ts)
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

        ts = StaticScenario(name="scenario", test_runs=[dep_tr, main_tr])
        runner = MyRunner("run", system, ts)
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

        ts = StaticScenario(name="scenario", test_runs=[post_comp_tr, post_init_tr, main_tr])
        runner = MyRunner("run", system, ts)
        runner.run()

        assert len(runner.active_jobs) == 0
        assert len(runner.completed_jobs) == 3
        assert runner.completed_jobs[main_tr.name].id == 0
        assert runner.completed_jobs[main_tr.name].test_run.name == main_tr.name
        assert runner.completed_jobs[post_init_tr.name].id == 1
        assert runner.completed_jobs[post_init_tr.name].test_run.name == post_init_tr.name
        assert runner.completed_jobs[post_comp_tr.name].id == 2
        assert runner.completed_jobs[post_comp_tr.name].test_run.name == post_comp_tr.name
