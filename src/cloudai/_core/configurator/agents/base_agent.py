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
from typing import Any, Dict

from cloudai._core.test_scenario import TestRun


class BaseAgent(ABC):
    """
    Base class for all agents in the CloudAI framework.

    Provides a unified interface and parameter management for action spaces.
    Automatically infers parameter types from TestRun's cmd_args.
    """

    def __init__(self, test_run: TestRun):
        """
        Initialize the agent with the TestRun object.

        Args:
            test_run (TestRun): The TestRun object containing cmd_args and test state.
        """
        self.test_run = test_run
        self.action_space = self.extract_action_space(test_run.test.cmd_args)

    def extract_action_space(self, cmd_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the action space from cmd_args by inferring parameter types.

        Args:
            cmd_args (Dict[str, Any]): The command arguments from TestRun.

        Returns:
            Dict[str, Any]: Action space defined with inferred parameter types.
        """
        action_space = {}

        for key, value in cmd_args.items():
            self._process_value(action_space, key, value)

        return action_space

    def _process_value(self, action_space: Dict[str, Any], key: str, value: Any) -> None:
        if isinstance(value, list):
            self._process_list(action_space, key, value)
        elif isinstance(value, (int, float)):
            action_space[key] = {"type": "fixed", "value": value}
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                full_key = f"{key}.{sub_key}"
                self._process_value(action_space, full_key, sub_value)

    def _process_list(self, action_space: Dict[str, Any], key: str, value: list) -> None:
        if all(isinstance(v, int) for v in value):
            action_space[key] = {"type": "int", "range": (min(value), max(value))}
        elif all(isinstance(v, float) for v in value):
            action_space[key] = {"type": "float", "range": (min(value), max(value))}
        else:
            action_space[key] = {"type": "categorical", "categories": value}

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the agent with additional settings.

        Args:
            config (Dict[str, Any]): Configuration settings for the agent.
        """
        pass

    @abstractmethod
    def select_action(self) -> Dict[str, Any]:
        """
        Select an action from the action space.

        Returns:
            Dict[str, Any]: A dictionary mapping action keys to selected values.
        """
        pass

    @abstractmethod
    def update_policy(self, feedback: Dict[str, Any]) -> None: # noqa: F841
        """
        Update the agent state based on feedback from the environment.

        Args:
            feedback (Dict[str, Any]): Feedback information from the environment.
        """
        pass
