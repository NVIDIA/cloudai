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

from typing import Dict, Tuple

import numpy as np

from src.cloudai._core.cloudai_gym import CloudAIGym


class ExampleEnv(CloudAIGym):
    """A sample environment that extends CloudAIGym."""

    def __init__(self, max_steps: int = 10):
        """
        Initialize the environment.

        Args:
            max_steps (int): Maximum number of steps allowed in the environment.
        """
        # Define the action and observation structures
        action_space = {
            "num_cores": (0, 15),  # Integer range: [0, 14]
            "freq": (0.5, 3.0),  # Float range: [0.5, 3.0]
            "mem_type": (0, 2),  # Integer range: [0, 2]
            "mem_size": (0, 64),  # Integer range: [0, 64]
        }
        observation_space = {
            "energy": (0.0, 1.0),
            "area": (0.0, 1.0),
            "latency": (0.0, 1.0),
        }

        super().__init__(action_space, observation_space, max_steps)

        # Additional environment attributes
        self.ideal = np.array([4, 2.0, 1, 32])  # Ideal values
        self.reset()

    def is_valid_action(self, action: dict) -> bool:
        """
        Validate whether the action falls within the defined bounds.

        Args:
            action (dict): A dictionary representing the action.

        Returns:
            bool: True if valid, False otherwise.

        """
        for key, value in action.items():
            if key not in self.action_space:
                return False
            lower, upper = self.action_space[key]
            if not (lower <= value <= upper):
                return False
        return True

    def step(self, action: dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute a step in the environment."""
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")

        # Unpack action components
        num_cores = action["num_cores"]
        freq = action["freq"]
        mem_type = action["mem_type"]
        mem_size = action["mem_size"]

        # Update state
        self.energy += num_cores * 1 + freq * 2 + mem_size * 3
        self.area += num_cores * 2 + freq * 3 + mem_size * 1
        self.latency += num_cores * 3 + freq * 3 + mem_size * 1

        # Update observation
        self.observation = np.array([self.energy, self.area, self.latency])

        # Calculate reward as negative L2-norm to the ideal values
        action_array = np.array([num_cores, freq, mem_type, mem_size])
        reward: float = -float(np.linalg.norm(action_array - self.ideal))

        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.observation, reward, done, {}

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to its initial state."""
        self.energy = 0
        self.area = 0
        self.latency = 0
        self.current_step = 0
        self.observation = np.array([self.energy, self.area, self.latency])
        return self.observation, {}

    def render(self) -> None:
        """Render the current state of the environment."""
        print(f"Energy: {self.energy}, Area: {self.area}, Latency: {self.latency}")
