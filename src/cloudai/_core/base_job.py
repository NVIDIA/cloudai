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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .system import System
from .test import Test


class BaseJob(ABC):
    """
    Base class for representing a job created by executing a test.

    Attributes
        mode (str): The mode of the job (e.g., 'run', 'dry-run').
        system (System): The system in which the job is running.
        test (Test): The test instance associated with this job.
        output_path (Path): The path where the job's output is stored.
        terminated_by_dependency (bool): Flag to indicate if the job was terminated due to a dependency.
    """

    def __init__(self, mode: str, system: System, test: Test, output_path: Path):
        """
        Initialize a BaseJob instance.

        Args:
            mode (str): The mode of the job (e.g., 'run', 'dry-run').
            system (System): The system in which the job is running.
            test (Test): The test instance associated with the job.
            output_path (Path): The path where the job's output is stored.
        """
        self.mode = mode
        self.system = system
        self.test = test
        self.output_path = output_path
        self.terminated_by_dependency = False

    @abstractmethod
    def get_id(self) -> Union[str, int]:
        """
        Abstract method to retrieve the unique identifier of the job.

        Returns
            Union[str, int]: The unique identifier of the job.
        """
        pass

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
        self.test.current_iteration += 1

    def __repr__(self) -> str:
        """
        Return a string representation of the BaseJob instance.

        Returns
            str: String representation of the job.
        """
        return f"BaseJob(mode={self.mode}, system={self.system.name}, test={self.test.name})"
