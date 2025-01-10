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

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai.systems import SlurmSystem


class CloudAIGymEnv(gym.Env):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(self, test_run: TestRun, system: SlurmSystem, test_scenario: TestScenario):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            system (SlurmSystem): The system configuration for running the tests.
            test_scenario (TestScenario): The test scenario configuration.
        """
        super(CloudAIGymEnv, self).__init__()
        self.test_run = test_run

        self.action_space = self.extract_action_space(self.test_run.test.cmd_args)
        self.observation_space = self.define_observation_space()

    def extract_action_space(self, cmd_args: dict) -> spaces.Dict:
        """
        Extract the action space from the cmd_args dictionary.

        Args:
            cmd_args (dict): The command arguments dictionary from the TestRun object.

        Returns:
            spaces.Dict: A dictionary containing the action space variables and their feasible values.
        """
        action_space = {}
        for key, value in cmd_args.items():
            if isinstance(value, list):
                action_space[key] = spaces.Discrete(len(value))
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        action_space[f"{key}.{sub_key}"] = spaces.Discrete(len(sub_value))
        return spaces.Dict(action_space)

    def define_observation_space(self) -> spaces.Space:
        """
        Define the observation space for the environment.

        Returns:
            spaces.Space: The observation space.
        """
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and reinitialize the TestRun.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
            options (Optional[dict]): Additional options for reset.

        Returns:
            Tuple: A tuple containing:
                - observation (np.ndarray): Initial observation.
                - info (dict): Additional info for debugging.
        """
        super().reset(seed=seed, options=options)
        self.test_run.current_iteration = 0
        observation = np.array([0.0], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one step in the environment.

        Args:
            action (np.ndarray): Action chosen by the agent.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): Updated system state.
                - reward (float): Reward for the action taken.
                - done (bool): Whether the episode is done.
                - info (dict): Additional info for debugging.
        """
        observation = self.get_observation(action)
        reward = 0.0
        done = False
        info = {}

        return observation, reward, done, info

    def render(self, mode="human"):
        """
        Render the current state of the TestRun.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def compute_reward(self) -> float:
        """
        Compute a reward based on the TestRun result.

        Returns:
            float: Reward value.
        """
        return 0.0

    def get_observation(self, action) -> np.ndarray:
        """
        Get the observation from the TestRun object.

        Returns:
            np.ndarray: A scalar value representing the observation.
        """
        obs = action * 0.5
        return np.array([obs], dtype=np.float32)
