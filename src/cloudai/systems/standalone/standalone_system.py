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

import logging
import os

from cloudai.core import BaseJob, System
from cloudai.util import CommandShell


class StandaloneSystem(System):
    """
    Class representing a Standalone system.

    This class is used for systems that execute commands directly without a job scheduler.
    """

    scheduler: str = "standalone"
    monitor_interval: int = 1
    cmd_shell: CommandShell = CommandShell()

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
        # Poll the handle when we own it. Shelling out to `ps` spawns two processes per
        # check and the check runs on every monitor tick, so on a fast backend it cost more
        # than the job being waited for. Polling also reaps the child, so a finished process
        # cannot linger as a zombie and keep reading as running.
        process = getattr(job, "process", None)
        if process is not None:
            is_running = process.poll() is None
            logging.debug(f"Job {job.id} running status: {is_running}")
            return is_running

        # No handle: the job was not launched by this process. Probe with signal 0, which
        # checks for the pid without delivering anything.
        try:
            os.kill(int(job.id), 0)
        except (ProcessLookupError, PermissionError, TypeError, ValueError):
            logging.debug(f"Job {job.id} running status: False")
            return False
        logging.debug(f"Job {job.id} running status: True")
        return True

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
        try:
            pid = int(str(job.id).strip())
        except ValueError:
            logging.warning(
                "Skipping termination for standalone job %s because it is not a valid process ID.",
                job.id,
            )
            return

        if pid <= 0:
            logging.warning(
                "Skipping termination for standalone job %s because it does not reference a launched process.",
                job.id,
            )
            return

        cmd = f"kill -9 {pid}"
        logging.debug(f"Executing termination command for job {pid}: {cmd}")
        self.cmd_shell.execute(cmd)
