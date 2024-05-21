# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional, cast

from cloudai.runner.core import BaseJob, BaseRunner, Runner
from cloudai.schema.core import System, Test, TestScenario
from cloudai.schema.system import SlurmSystem
from cloudai.util import CommandShell

from .slurm_job import SlurmJob


@Runner.register("slurm")
class SlurmRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Slurm.

    This class is responsible for executing and managing tests in a Slurm
    environment. It extends the BaseRunner class, implementing the abstract
    methods to work with Slurm jobs.

    Attributes
        slurm_system (SlurmSystem): This attribute is a casted version of the `system`
                                    attribute to `SlurmSystem` type, ensuring that
                                    Slurm-specific properties and methods are
                                    accessible.
        cmd_shell (CommandShell): An instance of CommandShell for executing
                                  system commands.
        Inherits all other attributes from the BaseRunner class.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario) -> None:
        """
        Initialize the SlurmRunner.

        Args:
            mode (str): The operation mode ('dry-run', 'run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.
        """
        super().__init__(mode, system, test_scenario)
        self.slurm_system: SlurmSystem = cast(SlurmSystem, system)
        self.cmd_shell = CommandShell()

    def _submit_test(self, test: Test) -> Optional[SlurmJob]:
        """
        Submit a test for execution on Slurm and returns a SlurmJob.

        Args:
            test (Test): The test to be executed.

        Returns:
            Optional[SlurmJob]: A SlurmJob object if the test execution is
                                successful, None otherwise.
        """
        self.logger.info(f"Running test: {test.section_name}")
        job_output_path = self.get_job_output_path(test)
        exec_cmd = test.gen_exec_command(job_output_path)
        self.logger.info(f"Executing command for test {test.section_name}: {exec_cmd}")
        job_id = None
        if self.mode == "run":
            stdout, stderr = self.cmd_shell.execute(exec_cmd).communicate()
            job_id = test.get_job_id(stdout, stderr)
        else:
            job_id = 0
        return SlurmJob(job_id, test) if job_id is not None else None

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if the specified job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        if self.mode == "dry-run":
            return True
        return self.slurm_system.is_job_running(job.id)

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a Slurm job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        if self.mode == "dry-run":
            return True
        s_job = cast(SlurmJob, job)
        return self.slurm_system.is_job_completed(s_job.id)

    def kill_job(self, job: BaseJob) -> None:
        """
        Terminate a Slurm job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        s_job = cast(SlurmJob, job)
        self.slurm_system.scancel(s_job.id)
