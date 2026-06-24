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
import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_agent import RewardOverrides
from .base_gym import BaseGym
from .env_params import CsvSink, EnvParamsObserver, EnvParamsSink, StepObserver


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    env_params: dict[str, Any] = dataclasses.field(default_factory=dict)


class CloudAIGymEnv(BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(self, test_run: TestRun, runner: BaseRunner, rewards: RewardOverrides):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.trajectory: dict[int, list[TrajectoryEntry]] = {}
        self._env_sink: EnvParamsSink | None = None
        self.observers: List[StepObserver] = self._build_observers()
        super().__init__()

    def _build_observers(self) -> List[StepObserver]:
        """
        Construct the per-step observers implied by the TestDefinition.

        Workloads opt in to env_params via a TOML ``[env_params.<name>]`` block;
        an empty mapping yields no observers and zero overhead.
        """
        observers: List[StepObserver] = []
        if self.test_run.test.env_params:
            seed = int((self.test_run.test.agent_config or {}).get("random_seed", 0))
            self._env_sink = CsvSink(self._env_csv_path())
            observers.append(EnvParamsObserver(self.test_run.test.env_params, self.test_run.test.cmd_args, seed))
        return observers

    def _env_csv_path(self) -> Path:
        """``env.csv`` lives alongside ``trajectory.csv`` so a plain ``merge`` joins them."""
        return self.trajectory_file_path.parent / "env.csv"

    def define_action_space(self) -> Dict[str, list[Any]]:
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
            lazy.np.random.seed(seed)
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
        self.test_run.increment_step()
        self.test_run = self.test_run.apply_params_set(action)

        for observer in self.observers:
            observer.before_step(self.test_run)

        # Overlay this trial's sampled env_params onto cmd_args so the workload actually
        # runs with the sampled values - the env-side twin of apply_params_set(action).
        # Sampling, persistence (env.csv), and the trajectory cache key are handled
        # separately; this is the single, workload-agnostic injection point. Keys are
        # validated to be cmd_args fields at TestDefinition build time; we re-filter to
        # those fields here so a programmatically-built run can't inject unknown attrs.
        if self.test_run.current_env_params:
            cmd_args = self.test_run.test.cmd_args
            fields = getattr(type(cmd_args), "model_fields", {})
            overlay = {k: v for k, v in self.test_run.current_env_params.items() if k in fields}
            if overlay:
                self.test_run.test.cmd_args = cmd_args.model_copy(update=overlay)

        cached_result = self.get_cached_trajectory_result(action)
        if cached_result is not None:
            logging.info(
                "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                cached_result.reward,
                cached_result.step,
            )
            self.write_trajectory(
                TrajectoryEntry(
                    step=self.test_run.step,
                    action=action,
                    reward=cached_result.reward,
                    observation=cached_result.observation,
                    env_params=dict(self.test_run.current_env_params),
                )
            )
            for observer in self.observers:
                observer.after_step(self.test_run, cached_result.observation, cached_result.reward)
            return cached_result.observation, cached_result.reward, False, {}

        if not self.test_run.test.constraint_check(self.test_run, self.runner.system):
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], self.rewards.constraint_failure, True, {}

        new_tr = copy.deepcopy(self.test_run)
        new_tr.output_path = self.runner.get_job_output_path(new_tr)
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

        # Rebuilding/replacing test_run above can drop the per-trial env_params sample;
        # restore it so the trajectory entry, the cache key, and env.csv all record the
        # params the trial actually ran with.
        self.test_run.current_env_params = new_tr.current_env_params

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)

        self.write_trajectory(
            TrajectoryEntry(
                step=self.test_run.step,
                action=action,
                reward=reward,
                observation=observation,
                env_params=dict(self.test_run.current_env_params),
            )
        )

        for observer in self.observers:
            observer.after_step(self.test_run, observation, reward)

        return observation, reward, False, {}

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
            if v is METRIC_ERROR:
                v = self.rewards.metric_failure
            observation.append(v)
        return observation

    def write_trajectory(self, entry: TrajectoryEntry):
        """
        Append the entry to the in-memory cache and trajectory.csv (plus env.csv when declared).

        ``trajectory.csv`` and the ``env.csv`` projection are sunk from the same
        ``TrajectoryEntry`` here, so a trial that never produces an entry (e.g. a
        constraint failure returns before this call) lands in neither file and the
        two stay 1:1 step-aligned.
        """
        self.current_trajectory.append(entry)

        file_exists = self.trajectory_file_path.exists()
        logging.debug(f"Writing trajectory into {self.trajectory_file_path} (exists: {file_exists})")
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trajectory_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation"])
            writer.writerow([entry.step, entry.action, entry.reward, entry.observation])

        if self._env_sink is not None:
            # current_iteration can advance while this env instance is reused, so rebind the sink to
            # the current iteration's env.csv (alongside trajectory.csv) to keep the two 1:1 aligned.
            self._env_sink = CsvSink(self._env_csv_path())
            self._env_sink.write(entry.step, entry.env_params)

    @property
    def trajectory_file_path(self) -> Path:
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}" / "trajectory.csv"

    @property
    def current_trajectory(self) -> list[TrajectoryEntry]:
        return self.trajectory.setdefault(self.test_run.current_iteration, [])

    def get_cached_trajectory_result(self, action: Any) -> TrajectoryEntry | None:
        """
        Return a cached entry only when the full trial identity matches.

        Trial identity is ``(action, env_params)``: env-randomized parameters
        change the workload's behaviour, so a trial repeating the same action
        under a different ``env_params`` sample must miss and re-run. Empty
        env_params on both sides is the back-compat path for workloads that
        do not declare any ``[env_params.*]`` block.
        """
        current_env_params = getattr(self.test_run, "current_env_params", {}) or {}
        for entry in self.current_trajectory:
            if not self._values_match_exact(entry.action, action):
                continue
            entry_env = getattr(entry, "env_params", {}) or {}
            if self._values_match_exact(entry_env, current_env_params):
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
