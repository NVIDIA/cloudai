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
from typing import Union

from .system import System
from .test_scenario import TestRun


class BaseJob:
    """
    Base class for representing a job created by executing a test.

    Attributes
        id (Union[str, int]): The unique identifier of the job.
        mode (str): The mode of the job (e.g., 'run', 'dry-run').
        system (System): The system in which the job is running.
        test_run (TestRun): The TestRun instance associated with this job.
        output_path (Path): The path where the job's output is stored.
        terminated_by_dependency (bool): Flag to indicate if the job was terminated due to a dependency.
    """

    def __init__(self, mode: str, system: System, test_run: TestRun):
        """
        Initialize a BaseJob instance.

        Args:
            mode (str): The mode of the job (e.g., 'run', 'dry-run').
            system (System): The system in which the job is running.
            test_run (TestRun): The TestRun instance associated with this job.
        """
        self.id: Union[str, int] = 0
        self.mode: str = mode
        self.system: System = system
        self.test_run: TestRun = test_run
        self.output_path: Path = test_run.output_path
        self.terminated_by_dependency: bool = False

    def is_running(self) -> bool:
        """
        Check if the specified job is currently running.

        Returns
            bool: True if the job is running, False otherwise.
        """
        if self.mode == "dry-run":
            return True
        return self.system.is_job_running(self)

    def is_completed(self) -> bool:
        """
        Check if a job is completed.

        Returns
            bool: True if the job is completed, False otherwise.
        """
        if self.mode == "dry-run":
            return True
        return self.system.is_job_completed(self)

    def increment_iteration(self):
        """Increment the iteration count of the associated test."""
        self.test_run.test.current_iteration += 1

    def __repr__(self) -> str:
        """
        Return a string representation of the BaseJob instance.

        Returns
            str: String representation of the job.
        """
        return f"BaseJob(id={self.id}, mode={self.mode}, system={self.system.name}, test={self.test_run.test.name})"
