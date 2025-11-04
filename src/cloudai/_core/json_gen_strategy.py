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

import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from .system import System
from .test_scenario import TestRun


class JsonGenStrategy(ABC):
    """
    Abstract base class for generating Kubernetes job specifications based on system and test parameters.

    It specifies how to generate JSON job specifications based on system and test parameters.
    """

    def __init__(self, system: System, test_run: TestRun) -> None:
        self.system = system
        self.test_run = test_run

    def sanitize_k8s_job_name(self, job_name: str) -> str:
        """
        Sanitize the job name to ensure it follows Kubernetes naming rules.

        - Must be lowercase.
        - Can only contain alphanumeric characters, hyphens, and periods.
        - Must start and end with an alphanumeric character.
        - Must be at most 253 characters long.

        Args:
            job_name (str): The original job name to be sanitized.

        Returns:
            str: The sanitized job name that complies with Kubernetes naming rules.
        """
        sanitized_name = job_name.lower()
        sanitized_name = re.sub(r"[^a-z0-9.-]", "-", sanitized_name)
        sanitized_name = re.sub(r"^[^a-z0-9]+", "", sanitized_name)
        sanitized_name = re.sub(r"[^a-z0-9]+$", "", sanitized_name)
        return sanitized_name[:253]

    @abstractmethod
    def gen_json(self) -> Dict[Any, Any]:
        """
        Generate the Kubernetes job specification based on the given parameters.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            Dict[Any, Any]: The generated Kubernetes job specification in JSON format.
        """
        pass
