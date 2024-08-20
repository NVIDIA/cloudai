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
from pathlib import Path

from cloudai import BaseJob, System
from cloudai.util import CommandShell


class StandaloneSystem(System):
    """
    Represents a standalone system without a job scheduler.

    Attributes
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    def __init__(self, name: str, output_path: Path) -> None:
        """
        Initialize a StandaloneSystem instance.

        Args:
            name (str): Name of the standalone system.
            output_path (Path): Path to the output directory.
        """
        super().__init__(name, "standalone", output_path)
        self.cmd_shell = CommandShell()

    def __repr__(self) -> str:
        """
        Provide a string representation of the StandaloneSystem instance.

        Returns
            str: String representation of the standalone system including its name and scheduler type.
        """
        return f"StandaloneSystem(name={self.name}, scheduler={self.scheduler})"

    def update(self) -> None:
        """
        Update the standalone system's state.

        This method is not typically used in standalone systems but is required for interface consistency.
        """
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if a given standalone job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        command = f"ps -p {job.get_id()}"
        logging.debug(f"Checking job status with command: {command}")
        stdout = self.cmd_shell.execute(command).communicate()[0]

        # Check if the job's PID is in the ps output
        is_running = str(job.get_id()) in stdout
        logging.debug(f"Job {job.get_id()} running status: {is_running}")

        return is_running

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a given standalone job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        return not self.is_job_running(job)

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a standalone job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        cmd = f"kill -9 {job.get_id()}"
        logging.debug(f"Executing termination command for job {job.get_id()}: {cmd}")
        self.cmd_shell.execute(cmd)
