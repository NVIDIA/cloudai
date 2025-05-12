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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from cloudai import BaseJob, System
from cloudai.util import CommandShell


class LSFNodeObj(BaseModel):
    """Represents a node in the LSF system."""

    name: str
    state: str
    user: Optional[str] = None


class LSFGroup(BaseModel):
    """Represents a group of nodes within a queue."""

    model_config = ConfigDict(extra="forbid")
    name: str
    nodes: List[str]


class LSFQueue(BaseModel):
    """Represents a queue within the LSF system."""

    model_config = ConfigDict(extra="forbid")
    name: str
    groups: List[LSFGroup] = []
    lsf_nodes: List[LSFNodeObj] = Field(default_factory=list, exclude=True)


class LSFSystem(BaseModel, System):
    """
    Represents an LSF system.

    Attributes:
        name (str): Name of the system.
        install_path (Path): Path to the installation directory.
        output_path (Path): Path to the output directory.
        queues (List[LSFQueue]): List of queues in the system.
        account (Optional[str]): Account name for resource usage.
        global_env_vars (Dict[str, Any]): Global environment variables for the system.
        scheduler (str): Scheduler type, default is "lsf".
        project_name (Optional[str]): Project name associated with the system.
        default_queue (Optional[str]): The default queue for job submission.
        monitor_interval (int): Interval for monitoring jobs, in seconds.
        app (Optional[str]): Application name associated with the system.
        os_version (Optional[str]): Operating system version.
        cmd_shell (CommandShell): Command shell for executing system commands.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    install_path: Path
    output_path: Path
    queues: List[LSFQueue] = Field(default_factory=list)
    account: Optional[str] = None
    global_env_vars: Dict[str, Any] = {}
    scheduler: str = "lsf"
    project_name: Optional[str] = None
    default_queue: Optional[str] = None
    monitor_interval: int = 60
    app: Optional[str] = None
    os_version: Optional[str] = None
    cmd_shell: CommandShell = Field(default=CommandShell(), exclude=True)

    @field_serializer("install_path", "output_path")
    def _path_serializer(self, v: Path) -> str:
        return str(v)

    def update(self) -> None:
        """
        Update the system object for an LSF system.

        This method queries the current state of each node using the 'bhosts' and 'bjobs' commands.
        """
        bhosts_output, _ = self.fetch_command_output("bhosts")
        bjobs_output, _ = self.fetch_command_output("bjobs -u all")
        node_user_map = self.parse_bjobs_output(bjobs_output)
        self.parse_bhosts_output(bhosts_output, node_user_map)

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if a specified LSF job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        command = f"bjobs -noheader -o stat {job.id}"
        stdout, _ = self.cmd_shell.execute(command).communicate()
        return stdout.strip() == "RUN"

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a specified LSF job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        command = f"bjobs -noheader -o stat {job.id}"
        stdout, _ = self.cmd_shell.execute(command).communicate()
        return stdout.strip() in ["DONE", "EXIT"]

    def kill(self, job: BaseJob) -> None:
        """
        Terminate an LSF job.

        Args:
            job (BaseJob): The job to terminate.
        """
        self.cmd_shell.execute(f"bkill {job.id}")

    def fetch_command_output(self, command: str) -> Tuple[str, str]:
        """
        Execute a system command and return its output.

        Args:
            command (str): The command to execute.

        Returns:
            Tuple[str, str]: The stdout and stderr from the command execution.
        """
        logging.debug(f"Executing command: {command}")
        stdout, stderr = self.cmd_shell.execute(command).communicate()
        if stderr:
            logging.error(f"Error executing command '{command}': {stderr}")
        return stdout, stderr

    def parse_bjobs_output(self, bjobs_output: str) -> Dict[str, str]:
        """
        Parse the output of the `bjobs` command to map nodes to users.

        Args:
            bjobs_output (str): The output of the `bjobs -u all` command.

        Returns:
            Dict[str, str]: A dictionary mapping node names to user names.
        """
        node_user_map = {}
        for line in bjobs_output.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            job_id, user, _, _, _, exec_host = parts[:6]
            if exec_host not in node_user_map:
                node_user_map[exec_host] = user
        return node_user_map

    def parse_bhosts_output(self, bhosts_output: str, node_user_map: Dict[str, str]) -> None:
        """
        Parse the output of the `bhosts` command and update the system's node information.

        Args:
            bhosts_output (str): The output of the `bhosts` command.
            node_user_map (Dict[str, str]): A dictionary mapping node names to user names.
        """
        self.queues = []
        queue_map = {}

        for line in bhosts_output.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            node_name, status, _, _, _, queue_name = parts[:6]

            if queue_name not in queue_map:
                queue_map[queue_name] = LSFQueue(name=queue_name)
            queue = queue_map[queue_name]

            user = node_user_map.get(node_name)
            node = LSFNodeObj(name=node_name, state=status, user=user)
            queue.lsf_nodes.append(node)

        self.queues = list(queue_map.values())
