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

import asyncio
import csv
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..runner import Runner
from ..test_scenario import METRIC_ERROR, TestRun
from .base_gym import BaseGym


class CloudAIGymEnv(BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(self, test_run: TestRun, runner: Runner):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (Runner): The runner object to execute jobs.
        """
        self.test_run = test_run
        self.runner = runner
        self.test_scenario = runner.runner.test_scenario
        self.max_steps = test_run.test.test_definition.agent_steps
        super().__init__()

    def define_action_space(self) -> Dict[str, Any]:
        """
        Define the action space for the environment.

        Returns:
            Dict[str, Any]: The action space.
        """
        action_space: Dict[str, Any] = {}
        cmd_args_dict = self.test_run.test.test_definition.cmd_args.model_dump()
        extra_env_vars_dict = self.test_run.test.test_definition.extra_env_vars

        combined_dict = {
            **{f"{key}": value for key, value in cmd_args_dict.items()},
            **{f"extra_env_vars.{key}": value for key, value in extra_env_vars_dict.items()},
        }

        self.populate_action_space("", combined_dict, action_space)

        return action_space

    def populate_action_space(self, prefix: str, d: dict, action_space: dict):
        for key, value in d.items():
            if isinstance(value, list):
                action_space[f"{prefix}{key}"] = value
            elif isinstance(value, dict):
                self.populate_action_space(f"{prefix}{key}.", value, action_space)
            else:
                action_space[f"{prefix}{key}"] = [value]

    def define_observation_space(self) -> list:
        """
        Define the observation space for the environment.

        Returns:
            list: The observation space.
        """
        return [0.0]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,  # noqa: Vulture
    ) -> Tuple[list, dict[str, Any]]:
        """
        Reset the environment and reinitialize the TestRun.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
            options (Optional[dict]): Additional options for reset.

        Returns:
            Tuple: A tuple containing:
                - observation (list): Initial observation.
                - info (dict): Additional info for debugging.
        """
        if seed is not None:
            np.random.seed(seed)
        self.test_run.current_iteration = 0
        observation = [0.0]
        info = {}
        return observation, info

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
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                self.update_test_run_obj(
                    self.test_run.test.test_definition.extra_env_vars, key[len("extra_env_vars.") :], value
                )
            else:
                self.update_test_run_obj(self.test_run.test.test_definition.cmd_args, key, value)

        if not self.test_run.test.test_definition.constraint_check:
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], -1.0, True, {}
        logging.info(f"Running step {self.test_run.current_iteration} with action {action}")
        asyncio.run(self.runner.run())

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)
        done = False
        info = {}

        self.write_trajectory(self.test_run.current_iteration, action, reward, observation)

        return observation, reward, done, info

    def render(self, mode: str = "human"):
        """
        Render the current state of the TestRun.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
        """
        if seed is not None:
            np.random.seed(seed)

    def compute_reward(self, observation: list) -> float:
        """
        Compute a reward based on the TestRun result.

        Args:
            observation (list): The observation list containing the average value.

        Returns:
            float: Reward value.
        """
        if observation and observation[0] != 0:
            return 1.0 / observation[0]
        return 0.0

    def get_observation(self, action: Any) -> list:
        """
        Get the observation from the TestRun object.

        Args:
            action (Any): Action taken by the agent.

        Returns:
            list: The observation.
        """
        v = self.test_run.get_metric_value(self.runner.runner.system, self.test_run.test.test_definition.agent_metric)
        if v == METRIC_ERROR:
            return [-1.0]
        return [v]

    def update_test_run_obj(self, obj: Any, attr_path: str, value: Any) -> None:
        """Update a nested attribute of an object."""
        attrs = attr_path.split(".")
        for attr in attrs[:-1]:
            if isinstance(obj, dict):
                if attr in obj:
                    obj = obj[attr]
                else:
                    raise AttributeError(f"dict object has no key {attr!r}")
            elif hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise AttributeError(f"{type(obj).__name__!r} object has no attribute {attr!r}")
        if isinstance(obj, dict):
            obj[attrs[-1]] = value
        else:
            setattr(obj, attrs[-1], value)

    def write_trajectory(self, step: int, action: Any, reward: float, observation: list):
        """
        Write the trajectory to a CSV file.

        Args:
            step (int): The current step number.
            action (Any): The action taken by the agent.
            reward (float): The reward received for the action.
            observation (list): The observation after taking the action.
        """
        output_path = self.runner.runner.system.output_path / self.runner.runner.test_scenario.name
        subdir = next(output_path.iterdir())
        trajectory_file_path = subdir / f"{self.test_run.current_iteration}" / "trajectory.csv"

        file_exists = trajectory_file_path.exists()

        with open(trajectory_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation"])
            writer.writerow([step, action, reward, observation])
