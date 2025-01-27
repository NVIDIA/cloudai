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

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cloudai import ReportGenerator
from cloudai._core.configurator.base_gym import BaseGym
from cloudai._core.runner import Runner
from cloudai._core.test_scenario import TestRun


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
        super().__init__()

    def define_action_space(self) -> Dict[str, Any]:
        """
        Define the action space for the environment.

        Returns:
            Dict[str, Any]: The action space.
        """
        action_space = {}
        for key, value in self.test_run.test.cmd_args.items():
            if isinstance(value, list):
                action_space[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        action_space[f"{key}.{sub_key}"] = sub_value
        return action_space

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
        is_valid, ICI_mesh, DCN_mesh = self.test_run.test.test_definition.constraint_checker.validate(action)

        if not is_valid:
            return [0.0], -1.0, True, {"error": "Invalid action"}

        for key, value in action.items():
            self.update_nested_attr(self.test_run.test.test_definition.cmd_args, key, value)

        asyncio.run(self.runner.run())

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)
        done = False
        info = {}
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
        output_path = self.runner.runner.system.output_path / self.runner.runner.test_scenario.name

        generator = ReportGenerator(output_path)
        generator.generate_report(self.test_scenario)

        subdir = next(output_path.iterdir())
        report_file_path = subdir / f"{self.test_run.current_iteration}" / f"{self.test_run.step}"

        observation = self.parse_report(report_file_path)
        return observation

    def parse_report(self, output_path: Path) -> list:
        """
        Parse the generated report to extract the observation.

        Args:
            output_path (Path): The path to the runner's output.

        Returns:
            list: The extracted observation.
        """
        report_file_path = output_path / "report.txt"
        if not report_file_path.exists():
            return [-1.0]

        with open(report_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Average:"):
                    average_value = float(line.split(":")[1].strip())
                    return [average_value]
        return [-1.0]

    def update_nested_attr(self, obj, attr_path, value):
        """Update a nested attribute of an object."""
        attrs = attr_path.split(".")
        prefix = "Grok"
        if attrs[0] == prefix:
            attrs = attrs[1:]
        for attr in attrs[:-1]:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise AttributeError(f"{type(obj).__name__!r} object has no attribute {attr!r}")
        setattr(obj, attrs[-1], value)
