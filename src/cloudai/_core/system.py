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


class System(ABC):
    """
    Base class representing a generic system.

    Attributes
        name (str): Unique name of the system.
        scheduler (str): Type of scheduler used by the system, determining the specific subclass of System to be used.
        output_path (str): Path to the output directory.
        monitor_interval (int): Interval in seconds for monitoring jobs.
    """

    def __init__(
        self,
        name: str,
        scheduler: str,
        output_path: str,
        monitor_interval: int = 1,
    ) -> None:
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
        raise NotImplementedError("Subclasses must implement this method.")
