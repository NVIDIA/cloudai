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

import logging
from typing import cast

from cloudai import BaseJob, BaseRunner, JobIdRetrievalError, System, Test, TestScenario
from cloudai.systems import KubernetesSystem
from cloudai.util import CommandShell

from .slurm_job import KubernetesJob


class KubernetesRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Kubernetes.

    This class is responsible for executing and managing tests in a Kubernetes environment. It extends the BaseRunner class,
    implementing the abstract methods to work with Kubernetes jobs.

    Attributes
        slurm_system (KubernetesSystem): This attribute is a casted version of the `system` attribute to `KubernetesSystem` type,
            ensuring that Kubernetes-specific properties and methods are accessible.
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
        Inherits all other attributes from the BaseRunner class.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario) -> None:
        """
        Initialize the KubernetesRunner.

        Args:
            mode (str): The operation mode ('dry-run', 'run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.
        """
        super().__init__(mode, system, test_scenario)
        self.slurm_system: KubernetesSystem = cast(KubernetesSystem, system)
        self.cmd_shell = CommandShell()

    def _submit_test(self, test: Test) -> KubernetesJob:
        """
        Submit a test for execution on Kubernetes and returns a KubernetesJob.

        Args:
            test (Test): The test to be executed.

        Returns:
            KubernetesJob: A KubernetesJob object
        """
        return KubernetesJob()

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if the specified job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        return True

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a Kubernetes job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        return True

    def kill_job(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        pass
