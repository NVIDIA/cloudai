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

import os
import subprocess


class CommandShell:
    """
    A class responsible for executing shell commands using a specified shell executable.

    Attributes
        executable (str): The path to the shell executable used for running commands.
    """

    def __init__(self, executable: str = "/bin/bash"):
        """
        Initialize the CommandShell with a shell executable.

        Args:
            executable (str): The shell executable path. Defaults to "/bin/bash".

        Raises:
            FileNotFoundError: If the specified executable does not exist.
        """
        if not os.path.exists(executable):
            raise FileNotFoundError(f"Executable '{executable}' not found.")
        self.executable = executable

    def execute(self, command: str) -> subprocess.Popen:
        """
        Execute a shell command and return its process.

        Args:
            command (str): The command to be executed.

        Returns:
            subprocess.Popen: The process object for the executed command.

        Raises:
            subprocess.CalledProcessError: If command execution fails.
        """
        process = subprocess.Popen(
            command,
            shell=True,
            executable=self.executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
