# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Dict, Literal, Optional, Tuple

Proposal = tuple[int, dict[str, Any]]
"""Agent proposal: the agent's own proposal identity and the action, as returned by ``select_action``."""


@dataclasses.dataclass
class Trial:  # noqa: D101
    trial_index: int
    action: dict[str, Any]
    env_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    step: int | None = None
    status: Literal["pending", "completed", "failed", "cached", "rejected"] = "pending"
    observation: list = dataclasses.field(default_factory=list)
    reward: float = 0.0
    output_path: pathlib.Path | None = None

    @property
    def done(self) -> bool:
        return self.status == "rejected"

    def as_feedback(self) -> dict[str, Any]:
        if self.status == "pending":
            raise ValueError("Cannot provide feedback for a pending trial")
        return {
            "trial_index": self.trial_index,
            "action": self.action,
            "value": self.reward,
            "observation": self.observation,
            "done": self.done,
            "status": self.status,
            "step": self.step,
            "output_path": self.output_path,
            "env_params": self.env_params,
        }


class BaseGym(ABC):
    """Base class for CloudAI Gym environments."""

    _consumed_steps: int = 0

    def __init__(self):
        """Initialize the CloudAIGym environment."""
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()

    @abstractmethod
    def define_action_space(self) -> Dict[str, Any]:
        """
        Define the action space for the environment.

        Returns:
            Dict[str, Any]: The action space.
        """
        pass

    @abstractmethod
    def define_observation_space(self) -> list:
        """
        Define the observation space for the environment.

        Returns:
            list: The observation space.
        """
        pass

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,  # noqa: Vulture
    ) -> Tuple[list, dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
            options (Optional[dict]): Additional options for reset.

        Returns:
            Tuple: A tuple containing:
                - observation (list): Initial observation.
                - info (dict): Additional info for debugging.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action (Any): Action chosen by the agent.

        Returns:
            Tuple: A tuple containing:
                - observation (list): Updated system state.
                - reward (float): Reward for the action taken.
                - done (bool): Whether the episode is done.
                - info (dict): Additional info for debugging.
        """
        pass

    def evaluate_batch(self, proposals: Iterable[Proposal], max_steps: Optional[int] = None) -> list[Trial]:
        """Support step-only environments with serial batch evaluation."""
        results: list[Trial] = []
        pending = iter(proposals)
        while max_steps is None or self._consumed_steps < max_steps:
            proposal = next(pending, None)
            if proposal is None:
                break
            trial_index, action = proposal
            observation, reward, done, info = self.step(action)
            if not done:
                self._consumed_steps += 1
            results.append(
                Trial(
                    trial_index=trial_index,
                    action=action,
                    reward=reward,
                    observation=observation,
                    status="rejected" if done else "completed",
                    step=None if done else self._consumed_steps,
                    output_path=None,
                    env_params=info.get("env_params", {}),
                )
            )
        return results

    @abstractmethod
    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
        """
        pass
