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

from abc import abstractmethod

from .test_scenario import TestRun
from .test_template_strategy import TestTemplateStrategy


class CommandGenStrategy(TestTemplateStrategy):
    """
    Abstract base class defining the interface for command generation strategies across different system environments.

    It specifies how to generate execution commands based on system and test parameters.
    """

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
    def gen_srun_command(self, tr: TestRun) -> str:
        """
        Generate the Slurm srun command for a test based on the given parameters.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated Slurm srun command.
        """
        pass

    @abstractmethod
    def gen_srun_success_check(self, tr: TestRun) -> str:
        """
        Generate the Slurm success check command to verify if a test run was successful.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated command to check the success of the test run.
        """
        pass
