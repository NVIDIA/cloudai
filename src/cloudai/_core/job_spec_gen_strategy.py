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
from pathlib import Path
from typing import Any, Dict, List

from .test_template_strategy import TestTemplateStrategy


class JobSpecGenStrategy(TestTemplateStrategy):
    """
    Abstract base class defining the interface for job specification generation strategies.

    It specifies how to generate job specifications based on system and test parameters.
    """

    @abstractmethod
    def gen_job_spec(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        job_name: str,
        num_nodes: int,
        nodes: List[str],
    ) -> Any:
        """
        Generate the job specification based on the given parameters.

        Args:
            env_vars (Dict[str, str]): Environment variables for the test.
            cmd_args (Dict[str, str]): Command-line arguments for the test.
            extra_env_vars (Dict[str, str]): Additional environment variables.
            extra_cmd_args (str): Additional command-line arguments.
            output_path (Path): Path to the output directory.
            job_name (str): The name of the job.
            num_nodes (int): The number of nodes to be used for the test execution.
            nodes (List[str]): List of nodes for test execution, optional.

        Returns:
            Any: The generated job specification, which could be a command, dictionary,
                 or another format depending on the implementation.
        """
        pass
