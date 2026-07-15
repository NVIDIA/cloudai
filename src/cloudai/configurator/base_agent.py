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

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field

from .base_gym import BaseGym


class RewardOverrides(BaseModel):
    """Optional reward and observation overrides for the agent."""

    model_config = ConfigDict(extra="forbid")

    constraint_failure: float = Field(
        default=-1.0,
        description="Reward when a constraint check fails.",
    )
    metric_failure: float = Field(
        default=-1.0,
        description="Observation value when a metric is missing or failed.",
    )


class BaseAgentConfig(BaseModel):
    """Base config class for all agents in the CloudAI framework."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int = 42
    start_action: Literal["random", "first"] = "random"
    rewards: RewardOverrides = Field(
        default_factory=RewardOverrides,
        description="Reward and observation overrides for the agent.",
    )
    trajectory_file_type: Literal["csv", "jsonl"] = "csv"


class BaseAgent(ABC):
    """
    Base class for all agents in the CloudAI framework.

    Provides a unified interface and parameter management for action spaces.
    """

    # Opt-in: agents that operate over a variable environment - one that changes per trial, whether
    # by env_params sampling (domain randomization) or a curriculum schedule - set this True. Default
    # False so env_params declared for an agent that cannot handle a varying env are rejected rather
    # than silently ignored.
    supports_variable_environment: bool = False

    def __init__(self, env: BaseGym, config: BaseAgentConfig):
        """
        Initialize the agent with the environment.

        Args:
            env (BaseGym): The environment instance for the agent.
            config (BaseAgentConfig): The agent configuration. Class is defined by `get_config_class` static method.
        """
        self.env = env
        self.config = config

        self.action_space = {}
        self.max_steps = 0

    @staticmethod
    @abstractmethod
    def get_config_class() -> type[BaseAgentConfig]:
        pass

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the agent with additional settings.

        Args:
            config (Dict[str, Any]): Configuration settings for the agent.
        """
        pass

    @abstractmethod
    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]] | None:
        """
        Select an action from the action space.

        Args:
            observation: Latest observation produced by the environment (``env.reset()`` on the
                first call, then the result of the prior ``env.step()``). Stateless agents may
                ignore this; observation-conditioned agents should use it.

        Returns:
            Tuple[int, Dict[str, Any]] | None: The current step index and a dictionary mapping action keys
                to selected values, or ``None`` to signal termination of the agent loop (``run()`` stops).
        """
        pass

    @abstractmethod
    def update_policy(self, _feedback: Dict[str, Any]) -> None:
        """
        Update the agent state based on feedback from the environment.

        Args:
            feedback (Dict[str, Any]): Feedback information from the environment.
        """
        pass

    def run(self) -> int:
        """
        Orchestrate this agent's exploration over ``self.env``.

        Default: a step loop driven by the dispatcher (``select_action`` →
        ``env.step`` → ``update_policy`` per trial). Agents that drive their
        own training loop override this method.

        Failure contract (``handle_dse_job`` consumes the result via
        ``err |= agent.run()``):

        - Return a non-zero code for *recoverable* failures (e.g. a workload run
          that failed but should not abort the rest of the sweep). The code is
          accumulated and the next ``TestRun`` still executes. Workload-level
          failures are already surfaced this way: ``CloudAIGymEnv.step`` maps a
          failed metric to ``rewards.metric_failure`` rather than raising, and
          agents with their own training loop should likewise catch training
          errors and return a non-zero code.
        - Raise for *unexpected* failures (framework/agent bugs). Exceptions
          propagate out of ``handle_dse_job`` and hard-fail the job so the bug
          is surfaced instead of masked as a penalizing reward.

        Returns:
            int: Process-style return code (``0`` success, non-zero recoverable failure).
        """
        observation, _ = self.env.reset()
        for _ in range(self.max_steps):
            result = self.select_action(observation=observation)
            if result is None:
                break
            step, action = result
            logging.info("Running step %s (of %s) with action %s", step, self.max_steps, action)
            prev_observation = observation
            observation, reward, done, *_ = self.env.step(action)
            self.update_policy(
                {
                    "trial_index": step,
                    "value": reward,
                    "observation": observation,
                    "prev_observation": prev_observation,
                    "action": action,
                    "done": done,
                }
            )
            logging.info(
                "Step %s: Observation: %s, Reward: %.4f",
                step,
                [round(obs, 4) for obs in observation],
                reward,
            )
        return 0
