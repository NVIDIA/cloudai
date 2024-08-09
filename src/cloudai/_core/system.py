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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_job import BaseJob


class System(ABC):
    """
    Base class representing a generic system.

    Attributes
        name (str): Unique name of the system.
        scheduler (str): Type of scheduler used by the system, determining the specific subclass of System to be used.
        output_path (str): Path to the output directory.
        monitor_interval (int): Interval in seconds for monitoring jobs.
    """

    def __init__(self, name: str, scheduler: str, output_path: str, monitor_interval: int = 1) -> None:
        """
        Initialize a System instance.

        Args:
            name (str): Name of the system.
            scheduler (str): Type of scheduler used by the system.
            output_path (str): Path to the output directory.
            monitor_interval (int): Interval in seconds for monitoring jobs.
        """
        self.name = name
        self.scheduler = scheduler
        self.output_path = output_path
        self.monitor_interval = monitor_interval

    def __repr__(self) -> str:
        """
        Provide a detailed string representation of the System instance, including all its attributes.

        Returns
            str: String representation of the system including name, scheduler, output_path, and monitor_interval.
        """
        return (
            f"System(name='{self.name}', scheduler='{self.scheduler}', output_path='{self.output_path}', "
            f"monitor_interval={self.monitor_interval})"
        )

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
