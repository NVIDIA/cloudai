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

import copy
import csv
import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_gym import BaseGym


class TrajectoryEntry(TypedDict):
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    status: Literal["executed", "cached"]
    source_step: int | None


class CloudAIGymEnv(BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(self, test_run: TestRun, runner: BaseRunner):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.trajectory: list[TrajectoryEntry] = []
        super().__init__()

    def define_action_space(self) -> dict[str, list[Any]]:
        return self.test_run.param_space

    @property
    def first_sweep(self) -> dict[str, Any]:
        """Builds a sweep using first elements of each explorable parameter."""
        return {k: v[0] for k, v in self.define_action_space().items()}

    def define_observation_space(self) -> list:
        """
        Define the observation space for the environment.

        Returns:
            list: The observation space.
        """
        return [0.0]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: Vulture
    ) -> tuple[list, dict[str, Any]]:
        """
        Reset the environment and reinitialize the TestRun.

        Args:
            seed (int | None): Seed for the environment's random number generator.
            options (dict | None): Additional options for reset.

        Returns:
            Tuple: A tuple containing:
                - observation (list): Initial observation.
                - info (dict): Additional info for debugging.
        """
        if seed is not None:
            lazy.np.random.seed(seed)
        self.test_run.current_iteration = 0
        observation = [0.0]
        info = {}
        return observation, info

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
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
        self.test_run = self.test_run.apply_params_set(action)

        cached_result = self.get_cached_trajectory_result(action)
        if cached_result is not None:
            self.write_trajectory(
                self.test_run.step,
                action,
                cached_result["reward"],
                cached_result["observation"],
                status="cached",
                source_step=cached_result["source_step"],
            )
            logging.info(
                "Retrieved cached result from trajectory with reward %s",
                cached_result["reward"],
            )
            return cached_result["observation"], cached_result["reward"], False, {}

        if not self.test_run.test.constraint_check(self.test_run, self.runner.system):
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], -1.0, True, {}

        self.test_run.output_path = self.runner.get_job_output_path(self.test_run)

        new_tr = copy.deepcopy(self.test_run)
        new_tr.output_path = self.test_run.output_path
        self.runner.test_scenario.test_runs = [new_tr]

        self.runner.shutting_down = False
        self.runner.jobs.clear()
        self.runner.testrun_to_job_map.clear()

        try:
            self.runner.run()
        except Exception as e:
            logging.error(f"Error running step {self.test_run.step}: {e}")

        if self.runner.test_scenario.test_runs and self.runner.test_scenario.test_runs[0].output_path.exists():
            self.test_run = self.runner.test_scenario.test_runs[0]
        else:
            self.test_run = copy.deepcopy(self.original_test_run)
            self.test_run.step = new_tr.step
            self.test_run.output_path = new_tr.output_path

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)

        self.write_trajectory(
            self.test_run.step,
            action,
            reward,
            observation,
            status="executed",
            source_step=None,
        )

        return observation, reward, False, {}

    def render(self, mode: str = "human"):
        """
        Render the current state of the TestRun.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def seed(self, seed: int | None = None):
        """
        Set the seed for the environment's random number generator.

        Args:
            seed (int | None): Seed for the environment's random number generator.
        """
        if seed is not None:
            lazy.np.random.seed(seed)

    def compute_reward(self, observation: list) -> float:
        """
        Compute a reward based on the TestRun result.

        Args:
            observation (list): The observation list containing the average value.

        Returns:
            float: Reward value.
        """
        return self.reward_function(observation)

    def get_observation(self, action: Any) -> list:
        """
        Get the observation from the TestRun object.

        Args:
            action (Any): Action taken by the agent.

        Returns:
            list: The observation.
        """
        all_metrics = self.test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")

        observation = []
        for metric in all_metrics:
            v = self.test_run.get_metric_value(self.runner.system, metric)
            if v == METRIC_ERROR:
                v = -1.0
            observation.append(v)
        return observation

    def write_trajectory(
        self,
        step: int,
        action: Any,
        reward: float,
        observation: list,
        status: Literal["executed", "cached"],
        source_step: int | None = None,
    ):
        """
        Append the trajectory to the CSV file and to the local attribute.

        Args:
            step (int): The current step number.
            action (Any): The action taken by the agent.
            reward (float): The reward received for the action.
            observation (list): The observation after taking the action.
            status (str): Whether the trajectory row was executed or retrieved from cache.
            source_step (int | None): The executed step that supplied the cached result.
        """
        entry: TrajectoryEntry = {
            "step": step,
            "action": action,
            "reward": reward,
            "observation": observation,
            "status": status,
            "source_step": source_step,
        }
        self.trajectory.append(entry)

        file_exists = self.trajectory_file_path.exists()
        logging.debug(f"Writing trajectory into {self.trajectory_file_path} (exists: {file_exists})")
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trajectory_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(list(TrajectoryEntry.__annotations__.keys()))
            writer.writerow(list(entry.values()))

    @property
    def trajectory_file_path(self) -> Path:
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}" / "trajectory.csv"

    def get_cached_trajectory_result(self, action: Any) -> TrajectoryEntry | None:
        for entry in self.trajectory:
            if entry["status"] == "executed" and self._values_match_exact(entry["action"], action):
                return entry

        return None

    @classmethod
    def _values_match_exact(cls, left: Any, right: Any) -> bool:
        if type(left) is not type(right):
            return False

        elif isinstance(left, dict):
            left_keys = set(left.keys())
            right_keys = set(right.keys())
            if left_keys != right_keys:
                return False

            return all(cls._values_match_exact(left[key], right[key]) for key in left_keys)

        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                return False

            for left_item, right_item in zip(left, right, strict=True):
                if not cls._values_match_exact(left_item, right_item):
                    return False

            return True

        else:
            return left == right
