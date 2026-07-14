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
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_agent import RewardOverrides
from .base_gym import BaseGym
from .env_params import EnvParams, ObsLeafDescriptor
from .trajectory import (
    CsvTrajectoryWriter,
    EnvParamsSample,
    JsonLinesTrajectoryWriter,
    Trajectory,
    TrajectoryEntry,
    TrajectoryWriter,
    TrialResult,
)


class CloudAIGymEnv(BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(
        self,
        test_run: TestRun,
        runner: BaseRunner,
        rewards: RewardOverrides,
        *,
        trajectory_file_type: Literal["csv", "jsonl"] = "jsonl",
    ):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
            trajectory_file_type: Format used to persist trajectory records.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.params: EnvParams | None = EnvParams.from_test(test_run.test)
        self.trajectory = self._new_trajectory(trajectory_file_type)
        super().__init__()

    @property
    def upcoming_trial(self) -> int:
        """
        Index of the next trial ``step`` will run.

        ``step`` increments ``test_run.step`` before it samples/runs, so at rest the counter
        holds the last-run trial and the next one is ``+ 1``. ``reset`` peeks this to report the
        regime the first ``step`` will apply; ``step`` advances into it. The ``+ 1`` offset is
        defined only here so the two paths cannot disagree.
        """
        return self.test_run.step + 1

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
            list: One float slot per agent metric (at least one), giving the correct shape
            for adapters that derive ``gymnasium.spaces.Box`` from this output.
        """
        return [0.0] * max(len(self.test_run.test.agent_metrics), 1)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
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
        del options
        if seed is not None:
            lazy.np.random.seed(seed)
        self.test_run.current_iteration = 0
        info: dict[str, Any] = {}
        if self.params is not None:
            info["env_params"] = self.params.sample(self.upcoming_trial)
        return self.define_observation_space(), info

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
        trial = self.upcoming_trial
        self.test_run.increment_step()
        # RNG lives in the env: sample here, then apply action + sample so the run and cache key see them.
        sampled_env_params = self.params.sample(trial) if self.params else {}
        info: dict[str, Any] = {"env_params": sampled_env_params} if self.params is not None else {}
        self.test_run = self.test_run.apply_params_set(action, env_params=sampled_env_params)

        cached_result = self.get_cached_trajectory_result(action, sampled_env_params)

        if cached_result is not None:
            cached_trial_result = cached_result.get(TrialResult)
            if cached_trial_result is None:
                raise ValueError(f"cached trajectory entry at step {cached_result.step} is missing TrialResult")
            logging.info(
                "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                cached_trial_result.reward,
                cached_result.step,
            )
            observation = list(cached_trial_result.observation)
            reward = cached_trial_result.reward
        else:
            if not self.test_run.test.constraint_check(self.test_run, self.runner.system):
                logging.info("Constraint check failed. Skipping step.")
                return [-1.0], self.rewards.constraint_failure, True, info

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

            observation = self.get_observation(action)
            reward = self.compute_reward(observation)

        optional_values: dict[str, object] = {}
        if self.params is not None:
            optional_values["env_params"] = dict(sampled_env_params)
        self.trajectory.append(
            step=self.test_run.step,
            action=dict(action),
            reward=reward,
            observation=observation,
            **optional_values,
        )

        return observation, reward, False, info

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

    def structured_observation_descriptors(self) -> Optional[Dict[str, ObsLeafDescriptor]]:
        """
        Per-leaf descriptors for the env_param regime, or ``None`` when none are declared.

        The flat observation (metrics) is unchanged; these describe the env_param leaves the
        ``GymnasiumAdapter`` merges with it into its structured observation ``spaces.Dict``.
        """
        return self.params.observation_descriptors() if self.params is not None else None

    def encode_env_params(self, env_params: dict[str, Any]) -> Dict[str, Any]:
        """
        Encode a queried regime (the env_params behind an observation) into named leaves.

        ``env_params`` is a ``{name: drawn value}`` regime (the ``info["env_params"]`` that
        ``reset``/``step`` report). The adapter pairs the result with the flat metrics observation.
        """
        return self.params.encode(env_params) if self.params is not None else {}

    def _new_trajectory(self, file_type: Literal["csv", "jsonl"]) -> Trajectory:
        writer: TrajectoryWriter
        if file_type == "csv":
            writer = CsvTrajectoryWriter(lambda: self.iteration_dir)
        elif file_type == "jsonl":
            writer = JsonLinesTrajectoryWriter(lambda: self.iteration_dir)
        else:
            raise ValueError(f"Invalid file type: {file_type}")

        if self.params is None:
            return Trajectory(writer=writer)

        return Trajectory(
            writer=writer,
            components=(EnvParamsSample,),
            identity=(EnvParamsSample,),
        )

    @property
    def iteration_dir(self) -> Path:
        """Per-iteration output directory containing the trajectory output."""
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}"

    @property
    def trajectory_file_path(self) -> Path:
        path = self.trajectory.output_path
        if path is None:
            raise RuntimeError("trajectory persistence is not configured")
        return path

    def get_cached_trajectory_result(self, action: Any, env_params: dict[str, Any]) -> TrajectoryEntry | None:
        """
        Return a cached entry only when the full trial identity matches.

        Trial identity is ``(action, env_params)``: env-randomized parameters
        change the workload's behaviour, so a trial repeating the same action
        under a different ``env_params`` sample must miss and re-run. Empty
        env_params on both sides is the back-compat path for workloads that
        do not declare any ``[env_params.*]`` block. The sample is passed in (a
        per-trial local owned by ``step``), exactly like ``action``.
        """
        if not env_params:
            return self.trajectory.find(action)
        return self.trajectory.find(action, env_params=dict(env_params))
