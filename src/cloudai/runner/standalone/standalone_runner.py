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

import logging
import subprocess
from typing import cast

from cloudai import BaseRunner, JobIdRetrievalError, System, TestRun, TestScenario
from cloudai._core.base_runner import NewBaseRunner
from cloudai._core.cases_iter import StaticCasesListIter
from cloudai.systems.standalone_system import StandaloneSystem
from cloudai.util import CommandShell

from .standalone_job import StandaloneJob


class StandaloneRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Standalone.

    Attributes
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario) -> None:
        """
        Initialize the StandaloneRunner.

        Args:
            mode (str): The operation mode ('run', 'dry-run').
            system (System): The system object.
            test_scenario (TestScenario): The test scenario to run.
        """
        super().__init__(mode, system, test_scenario)
        self.cmd_shell = CommandShell()

    def _submit_test(self, tr: TestRun) -> StandaloneJob:
        """
        Submit a test for execution on Standalone and returns a StandaloneJob.

        Args:
            tr (TestRun): The test run to be executed.

        Returns:
            StandaloneJob: A StandaloneJob object
        """
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        exec_cmd = tr.test.test_template.gen_exec_command(tr)
        logging.info(f"Executing command for test {tr.name}: {exec_cmd}")
        job_id = 0
        if self.mode == "run":
            pid = self.cmd_shell.execute(exec_cmd).pid
            job_id = tr.test.test_template.get_job_id(str(pid), "")
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(tr.name),
                    command=exec_cmd,
                    stdout="",
                    stderr="",
                    message="Failed to retrieve job ID from command output.",
                )
        return StandaloneJob(tr, id=job_id)


class NewStandaloneRunner(NewBaseRunner):
    """Standalone Runner."""

    def __init__(self, mode: str, system: StandaloneSystem, test_scenario: TestScenario):
        super().__init__(mode, system, test_scenario)
        self.system = cast(StandaloneSystem, system)
        self.test_scenario_iter = StaticCasesListIter(test_scenario)

        self.active_jobs: dict[str, StandaloneJob] = {}
        self.completed_jobs: dict[str, StandaloneJob] = {}

        self.running_procs: dict[int, subprocess.Popen] = {}

    def process_completed_jobs(self):
        for job in self.active_jobs.values():
            completed = job.id not in self.running_procs
            if job.id in self.running_procs:
                p = self.running_procs[job.id]
                if p.poll() is not None:
                    completed = True
                    del self.running_procs[job.id]
            if completed:
                logging.info(f"Job {job.id} completed, see {job.test_run.output_path}")
                self.completed_jobs[job.test_run.name] = job
                self.test_scenario_iter.on_completed(job.test_run, self)

    def clean_active_jobs(self):
        in_common = set(self.active_jobs.keys()) & set(self.completed_jobs.keys())
        for name in in_common:
            del self.active_jobs[name]

    def submit_one(self, tr: TestRun) -> None:
        command = tr.test.test_template.gen_exec_command(tr)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.running_procs[p.pid] = p
        logging.info(f"Running test {tr.name} as Job.id={p.pid}")
        job = StandaloneJob(tr, p.pid)
        self.active_jobs[tr.name] = job

    def kill_one(self, tr: TestRun) -> None:
        logging.info(f"Killing job {tr.name}")
        job = self.active_jobs.get(tr.name)
        if job and job.id in self.running_procs:
            p = self.running_procs[job.id]
            p.kill()
            del self.running_procs[job.id]
