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

from pathlib import Path
from typing import Dict, List, Optional


class JobContext:
    """
    Encapsulates all necessary parameters required to generate a job specification.

    Attributes
        env_vars (Dict[str, str]): Environment variables for the test.
        cmd_args (Dict[str, str]): Command-line arguments for the test.
        extra_env_vars (Dict[str, str]): Additional environment variables.
        extra_cmd_args (str): Additional command-line arguments.
        output_path (Path): Path to the output directory.
        job_name (Optional[str]): The name of the job, if applicable.
        num_nodes (int): The number of nodes to be used for the test execution.
        nodes (List[str]): List of nodes for test execution, optional.
    """

    def __init__(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        job_name: Optional[str],
        num_nodes: int,
        nodes: List[str],
    ):
        """
        Initialize a new JobContext instance with the provided parameters.

        Args:
            env_vars (Dict[str, str]): Environment variables for the test.
            cmd_args (Dict[str, str]): Command-line arguments for the test.
            extra_env_vars (Dict[str, str]): Additional environment variables.
            extra_cmd_args (str): Additional command-line arguments.
            output_path (Path): Path to the output directory.
            job_name (Optional[str]): The name of the job, if applicable.
            num_nodes (int): The number of nodes to be used for the test execution.
            nodes (List[str]): List of nodes for test execution, optional.
        """
        self.env_vars = env_vars
        self.cmd_args = cmd_args
        self.extra_env_vars = extra_env_vars
        self.extra_cmd_args = extra_cmd_args
        self.output_path = output_path
        self.job_name = job_name
        self.num_nodes = num_nodes
        self.nodes = nodes
