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

import itertools
from typing import Any, Dict, List, Tuple

from cloudai import BaseAgent

from .cloudai_gym import CloudAIGymEnv


class GridSearchAgent(BaseAgent):
    """
    Agent implementing a grid search over the action space.

    Iterates through all possible parameter combinations.
    """

    def __init__(self, env: CloudAIGymEnv):
        """
        Initialize the GridSearchAgent with the TestRun object.

        Args:
             env (CloudAIGymEnv): The environment instance to query the action space from.
        """
        self.action_space = env.define_action_space()
        self.env = env
        self.action_combinations = []
        self.index = 0
        self.configure(self.action_space)

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the grid search by precomputing all parameter combinations.

        Args:
            config (Dict[str, Any]): The action space to configure.
        """
        parameter_values = []
        for _, values in config.items():
            parameter_values.append(values)

        self.action_combinations = list(itertools.product(*parameter_values))
        self.max_steps = len(self.action_combinations)

    def get_all_combinations(self) -> List[Dict[str, Any]]:
        """
        Get all possible combinations of the action space parameters.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of parameters.
        """
        keys = list(self.action_space.keys())
        return [dict(zip(keys, combination, strict=True)) for combination in self.action_combinations]

    def select_action(self) -> Tuple[int, Dict[str, Any]]:
        """
        Select the next action from the grid.

        Returns:
            Tuple[int, Dict[str, Any]]: The current step and a dictionary mapping action keys to selected
            values.
        """
        action = dict(zip(self.action_space.keys(), self.action_combinations[self.index], strict=True))
        self.index += 1
        step = self.index
        return step, action

    def update_policy(self, _feedback: Dict[str, Any]) -> None:
        """
        Update the agent based on feedback (not used in grid search).

        Args:
            feedback (Dict[str, Any]): Feedback information from the environment.
        """
        # Grid search is stateless and does not rely on feedback.
        pass
