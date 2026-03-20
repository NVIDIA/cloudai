# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024, 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
from pathlib import Path


class CommandShell:
    """
    A class responsible for executing shell commands using a specified shell executable.

    Attributes
        executable (Path): The path to the shell executable used for running commands.
    """

    def __init__(self, executable: Path = Path("/bin/bash")):
        """Initialize the CommandShell with a shell executable."""
        self.executable = executable

    def execute(self, command: str) -> subprocess.Popen:
        """Execute a shell command and return its process."""
        if not self.executable.exists():
            raise FileNotFoundError(f"Executable '{self.executable}' not found.")

        process = subprocess.Popen(
            command,
            shell=True,
            executable=str(self.executable),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
