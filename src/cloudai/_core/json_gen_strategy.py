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
from typing import Any, Dict, List

from .test_template_strategy import TestTemplateStrategy


class JsonGenStrategy(TestTemplateStrategy):
    """
    Abstract base class for generating Kubernetes job specifications based on system and test parameters.

    It specifies how to generate JSON job specifications based on system and test parameters.
    """

    @abstractmethod
    def gen_json(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        job_name: str,
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[Any, Any]:
        """
        Generate the Kubernetes job specification based on the given parameters.

        Args:
            env_vars (Dict[str, str]): Environment variables for the job.
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            extra_env_vars (Dict[str, str]): Additional environment variables.
            extra_cmd_args (str): Additional command-line arguments.
            output_path (str): Path to the output directory.
            job_name (str): The name of the job.
            num_nodes (int): The number of nodes to be used for job execution.
            nodes (List[str]): List of nodes for job execution, optional.

        Returns:
            Dict[Any, Any]: The generated Kubernetes job specification in JSON format.
        """
        pass
