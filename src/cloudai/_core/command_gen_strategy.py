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

from .test_scenario import TestRun
from .test_template_strategy import TestTemplateStrategy


class CommandGenStrategy(TestTemplateStrategy, ABC):
    """Abstract base class defining the interface for command generation strategies across different systems."""

    TEST_RUN_DUMP_FILE_NAME: str = "test-run.toml"

    @abstractmethod
    def gen_exec_command(self, tr: TestRun) -> str:
        """
        Generate the execution command for a test based on the given parameters.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated execution command.
        """
        pass

    @abstractmethod
    def store_test_run(self, tr: TestRun) -> None:
        """
        Store the test run information in output folder.

        Only at command generation time, CloudAI has all the information to store the test run.

        Args:
            tr (TestRun): The test run object to be stored.
        """
        pass
