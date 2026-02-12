# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Optional, Tuple

from cloudai.models.agent_config import AgentConfig

from .base_gym import BaseGym


class BaseAgent(ABC):
    """
    Base class for all agents in the CloudAI framework.

    Provides a unified interface and parameter management for action spaces.
    Automatically infers parameter types from TestRun's cmd_args.
    """

    config: Optional[AgentConfig] = None

    def __init__(self, env: BaseGym):
        """
        Initialize the agent with the environment.

        Args:
            env (BaseGym): The environment instance for the agent.
        """
        self.action_space = {}
        self.max_steps = 0

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the agent with additional settings.

        Args:
            config (Dict[str, Any]): Configuration settings for the agent.
        """
        pass

    @abstractmethod
    def select_action(self) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action from the action space.

        Returns:
            Tuple[int, Dict[str, Any]]: The current step index and a dictionary mapping action keys to selected values.
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
