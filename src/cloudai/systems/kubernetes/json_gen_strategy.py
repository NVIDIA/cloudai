# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import toml

from cloudai.core import JobStatusResult, System, TestRun

from .kubernetes_job import KubernetesJob


class JsonGenStrategy(ABC):
    """
    Abstract base class for generating Kubernetes job specifications based on system and test parameters.

    It specifies how to generate JSON job specifications based on system and test parameters.
    """

    TEST_RUN_DUMP_FILE_NAME: str = "test-run.toml"

    def start(self) -> KubernetesJob: ...
    def stop(self) -> None: ...
    def status(self) -> JobStatusResult: ...
    def is_running(self) -> bool: ...
    def is_completed(self) -> bool: ...

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
        sanitized_name = re.sub(r"[^a-z0-9-]", "-", sanitized_name)
        sanitized_name = re.sub(r"^[^a-z0-9]+", "", sanitized_name)
        sanitized_name = sanitized_name[:253]
        sanitized_name = re.sub(r"[^a-z0-9]+$", "", sanitized_name)

        if not sanitized_name:
            raise ValueError(f"'{job_name}' cannot be sanitized to a valid Kubernetes job name.")

        return sanitized_name

    def store_test_run(self) -> None:
        from cloudai.models.scenario import TestRunDetails

        test_cmd, srun_cmd = ("", "n/a")
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=srun_cmd)
            toml.dump(trd.model_dump(), f)

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

    def start_job(self) -> KubernetesJob:
        job_name = self.create_job()
        job = KubernetesJob(self.test_run, id=job_name, name=job_name, kind="job_kind")
        return job

    def _create_job(self) -> str: ...
    def _is_job_observable(self) -> bool: ...

    @final
    def create_job(self, timeout: int = 60) -> str:
        job_name = self._create_job()

        # Wait for the job to be observable by Kubernetes
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_job_observable():
                return job_name
            time.sleep(self.system.monitor_interval)

        raise TimeoutError(f"Job '{job_name}' was not observable within {timeout} seconds.")

    def delete_job(self) -> None: ...
