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


import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .installables import Installable

if TYPE_CHECKING:
    from .base_job import BaseJob


class System(ABC, BaseModel):
    """Base class representing a generic system."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    scheduler: str
    install_path: Path
    output_path: Path
    hf_home_path: Path = Field(default_factory=lambda data: data["install_path"] / "huggingface")
    global_env_vars: dict[str, Any] = Field(default_factory=dict)
    monitor_interval: int = 1

    @abstractmethod
    def update(self) -> None:
        """
        Update the system's state.

        Raises
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        error_message = (
            "System update method is not implemented. All subclasses of the System class must implement the "
            "'update' method to ensure the system's state can be refreshed as needed."
        )
        logging.error(error_message)
        raise NotImplementedError(error_message)

    @abstractmethod
    def is_job_running(self, job: "BaseJob") -> bool:
        """
        Check if a given job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.

        Raises:
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        error_message = (
            "Job running status check method is not implemented. All subclasses of the System class must implement the"
            " 'is_job_running' method to determine whether a job is currently active."
        )
        logging.error(error_message)
        raise NotImplementedError(error_message)

    @abstractmethod
    def is_job_completed(self, job: "BaseJob") -> bool:
        """
        Check if a given job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.

        Raises:
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        error_message = (
            "Job completion status check method is not implemented. All subclasses of the System class must implement "
            "the 'is_job_completed' method to determine whether a job has finished execution."
        )
        logging.error(error_message)
        raise NotImplementedError(error_message)

    @abstractmethod
    def kill(self, job: "BaseJob") -> None:
        """
        Terminate a given job.

        Args:
            job (BaseJob): The job to be terminated.

        Raises:
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        error_message = (
            "Job termination method is not implemented. All subclasses of the System class must implement the 'kill' "
            "method to terminate a job that is currently running."
        )
        logging.error(error_message)
        raise NotImplementedError(error_message)

    def system_installables(self) -> list[Installable]:
        return []
