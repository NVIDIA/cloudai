# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod

from .system import System
from .test_scenario import TestRun
from .test_template_strategy import TestTemplateStrategy


class CommandGenStrategy(TestTemplateStrategy, ABC):
    """Abstract base class defining the interface for command generation strategies across different systems."""

    TEST_RUN_DUMP_FILE_NAME: str = "test-run.toml"

    def __init__(self, system: System, test_run: TestRun) -> None:
        super().__init__(system)
        self.test_run = test_run
        self._final_env_vars: dict[str, str | list[str]] = {}

    @abstractmethod
    def gen_exec_command(self) -> str:
        """
        Generate the execution command for a test based on the given parameters.

        Returns:
            str: The generated execution command.
        """
        pass

    @abstractmethod
    def store_test_run(self) -> None:
        """
        Store the test run information in output folder.

        Only at command generation time, CloudAI has all the information to store the test run.
        """
        pass

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        if not self._final_env_vars:
            final_env_vars = self.system.global_env_vars.copy()
            final_env_vars.update(self.test_run.test.extra_env_vars)
            self._final_env_vars = final_env_vars
        return self._final_env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        self._final_env_vars = value
