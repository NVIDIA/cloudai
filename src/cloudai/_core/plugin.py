#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List


class Plugin:
    """
    A class to represent plugins actions in a test scenario.

    Attributes
        commands (List[str]): List of shell commands to be executed in the plugin.
    """

    def __init__(self, commands: List[str]) -> None:
        """
        Initialize a Plugin instance.

        Args:
            commands (List[str]): List of commands to execute as part of the plugin.
        """
        self.commands = commands

    def __repr__(self) -> str:
        """Return a string representation of the Plugin instance."""
        return f"Plugin(commands={self.commands})"

    def execute(self) -> None:
        """Execute all commands in the plugin."""
        for command in self.commands:
            print(f"Executing command: {command}")
