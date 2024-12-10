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
from typing import Any, Dict, Tuple


class CloudAIGym(ABC):
    """A generic custom Gym environment for CloudAI."""

    def __init__(self, action_space: Any, observation_space: Any, max_steps: int = 100):
        """
        Initialize the environment.

        Args:
            action_space: Defines the set of valid actions for the environment.
            observation_space: Describes the space of possible observations (states).
            max_steps: Maximum number of steps in an episode.
        """
        self.action_space: Any = action_space
        self.observation_space: Any = observation_space
        self.max_steps: int = max_steps
        self.state: Any = None
        self.current_step: int = 0

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        pass

    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render the environment."""
        pass

    @abstractmethod
    def is_valid_action(self, action: Any) -> bool:
        """Contraint checks for the action."""
        pass
