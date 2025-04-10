# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List


class AbstractPinningStrategy(ABC):
    """Abstract base class for system-specific pinning strategies."""

    def __init__(self, cpus_per_node: int, num_tasks_per_node: int) -> None:
        self.cpus_per_node = cpus_per_node
        self.num_tasks_per_node = num_tasks_per_node

    @abstractmethod
    def get_pinning_flags(self) -> List[str]:
        """Return a list of pinning flags specific to the system."""
        pass


class NoOpPinningStrategy(AbstractPinningStrategy):
    """Pinning strategy that provides no extra flags."""

    def get_pinning_flags(self) -> List[str]:
        return []


class AwsPinningStrategy(AbstractPinningStrategy):
    """AWS-specific pinning strategy. Uses the number of CPUs per node to determine flags."""

    def get_pinning_flags(self) -> List[str]:
        cpus_per_task = self.cpus_per_node // self.num_tasks_per_node if self.num_tasks_per_node else self.cpus_per_node
        return ["--cpu-bind=verbose", f"--cpus-per-task={cpus_per_task}", "--hint=nomultithread"]


class AzurePinningStrategy(AbstractPinningStrategy):
    """Azure-specific pinning strategy. Uses the number of CPUs per node to determine flags."""

    def __init__(self, cpus_per_node: int, num_tasks_per_node: int) -> None:
        super().__init__(cpus_per_node, num_tasks_per_node)
        self.cores_per_task = cpus_per_node // num_tasks_per_node

    def get_pinning_flags(self) -> List[str]:
        masks = self._generate_masks()
        return [f'--cpu-bind=mask_cpu:"{",".join(masks)}"']

    def _generate_masks(self) -> List[str]:
        base_mask = (1 << self.cores_per_task) - 1
        return [f"{(base_mask << (i * self.cores_per_task)):x}" for i in range(self.num_tasks_per_node)]


def create_pinning_strategy(system_name: str, cpus_per_node: int, num_tasks_per_node: int) -> AbstractPinningStrategy:
    system_name = system_name.lower().strip()

    if re.search(r"\baws\b", system_name):
        return AwsPinningStrategy(cpus_per_node, num_tasks_per_node)
    elif re.search(r"\bazure\b", system_name):
        return AzurePinningStrategy(cpus_per_node, num_tasks_per_node)
    else:
        return NoOpPinningStrategy(cpus_per_node, num_tasks_per_node)
