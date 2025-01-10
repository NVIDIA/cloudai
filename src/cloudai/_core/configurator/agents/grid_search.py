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

import itertools
from typing import Any, Dict, List

from cloudai._core.configurator.agents.base_agent import BaseAgent
from cloudai._core.test_scenario import TestRun


class GridSearchAgent(BaseAgent):
    """
    Agent implementing a grid search over the action space.

    Iterates through all possible parameter combinations.
    """

    def __init__(self, test_run: TestRun):
        """
        Initialize the GridSearchAgent with the TestRun object.

        Args:
            test_run (TestRun): The TestRun object containing cmd_args and test state.
        """
        super().__init__(test_run)
        self.action_combinations = []
        self.index = 0

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the grid search by precomputing all parameter combinations.

        Args:
            config (Dict[str, Any]): Additional configuration settings (optional).
        """
        parameter_values = []
        for _key, param in self.action_space.items():
            if param["type"] == "int":
                parameter_values.append(range(param["range"][0], param["range"][1] + 1))
            elif param["type"] == "float":
                step = config.get("float_step", 0.1)
                parameter_values.append(
                    [
                        param["range"][0] + i * step
                        for i in range(int((param["range"][1] - param["range"][0]) / step) + 1)
                    ]
                )
            elif param["type"] == "categorical":
                parameter_values.append(param["categories"])

        self.action_combinations = list(itertools.product(*parameter_values))
        self.index = 0

    def get_all_combinations(self) -> List[Dict[str, Any]]:
        """
        Get all possible combinations of the action space parameters.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of parameters.
        """
        keys = list(self.action_space.keys())
        return [dict(zip(keys, combination)) for combination in self.action_combinations]

    def select_action(self) -> Dict[str, Any]:
        """
        Select the next action from the grid.

        Returns:
            Dict[str, Any]: A dictionary mapping action keys to selected values.
        """
        if self.index >= len(self.action_combinations):
            raise StopIteration("Grid search completed.")

        action = dict(zip(self.action_space.keys(), self.action_combinations[self.index]))
        self.index += 1
        return action

    def update_policy(self, feedback: Dict[str, Any]) -> None:
        """
        Update the agent based on feedback (not used in grid search).

        Args:
            feedback (Dict[str, Any]): Feedback information from the environment.
        """
        # Grid search is stateless and does not rely on feedback.
        pass
