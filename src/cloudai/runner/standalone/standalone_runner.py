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
from typing import cast

from cloudai import BaseJob, BaseRunner, JobIdRetrievalError, System, Test, TestScenario
from cloudai.util import CommandShell

from .standalone_job import StandaloneJob


class StandaloneRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Standalone.

    This class is responsible for executing and managing tests in a standalone environment. It extends the BaseRunner
    class, implementing the abstract methods to work with standalone jobs.

    Attributes
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
        Inherits all other attributes from the BaseRunner class.
    """

    def __init__(
        self,
        mode: str,
        system: System,
        test_scenario: TestScenario,
    ):
        """
        Initialize the StandaloneRunner with a system object, test scenario, and monitor interval.

        Args:
            mode (str): The operation mode ('run', 'dry-run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.
        """
        super().__init__(mode, system, test_scenario)
        self.cmd_shell = CommandShell()

    def _submit_test(self, test: Test) -> StandaloneJob:
        """
        Submit a test for execution on Standalone and returns a StandaloneJob.

        Args:
            test (Test): The test to be executed.

        Returns:
            StandaloneJob: A StandaloneJob object
        """
        logging.info(f"Running test: {test.section_name}")
        job_output_path = self.get_job_output_path(test)
        exec_cmd = test.gen_exec_command(job_output_path)
        logging.info(f"Executing command for test {test.section_name}: {exec_cmd}")
        job_id = 0
        if self.mode == "run":
            pid = self.cmd_shell.execute(exec_cmd).pid
            job_id = test.get_job_id(str(pid), "")
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(test.section_name),
                    command=exec_cmd,
                    stdout="",
                    stderr="",
                    message="Failed to retrieve job ID from command output.",
                )
        return StandaloneJob(job_id, test, job_output_path)

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if the specified job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        return True

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a standalone job is completed.

        Args:
            job (StandaloneJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        if self.mode == "dry-run":
            return True

        s_job = cast(StandaloneJob, job)
        command = f"ps -p {s_job.id}"
        logging.debug(f"Checking job status with command: {command}")
        stdout = self.cmd_shell.execute(command).communicate()[0]
        return str(s_job.id) not in stdout

    def kill_job(self, job: BaseJob):
        """
        Terminate a standalone job.

        Args:
            job (StandaloneJob): The job to be terminated.
        """
        s_job = cast(StandaloneJob, job)
        cmd = f"kill -9 {s_job.id}"
        logging.info(f"Executing termination command for job {s_job.id}: {cmd}")
        self.cmd_shell.execute(cmd)
