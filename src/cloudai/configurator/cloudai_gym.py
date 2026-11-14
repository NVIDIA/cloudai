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
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_agent import BaseAgentConfig, RewardOverrides
from .base_gym import BaseGym
from .env_params import EnvParams, ObsLeafDescriptor
from .trajectory import Trajectory

if TYPE_CHECKING:
    import pandas as pd


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
        max_consecutive_rejected_proposals: int = BaseAgentConfig().max_consecutive_rejected_proposals,
    ):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
            max_consecutive_rejected_proposals: Rejections allowed before consuming the current sweep step.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.max_consecutive_rejected_proposals = max_consecutive_rejected_proposals
        self._progress = test_run.step
        self._consecutive_rejections = 0
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.params: EnvParams | None = EnvParams.from_test(test_run.test)
        self.trajectory = Trajectory(iteration_dir=self.iteration_dir)
        super().__init__()

    def _ranked_dse_candidates(self) -> list[tuple[int, dict[str, str | int | float]]]:
        tr = self.original_test_run
        eligible_steps = {
            record.step for record in self.trajectory.records() if not record.ineligibility_reasons("measurement")
        }
        candidates = [
            row for row in self.trajectory.dataframe.to_dict(orient="records") if row["step"] in eligible_steps
        ]
        return [
            (int(row["step"]), {key: row[f"action.{key}"] for key in tr.param_space})
            for row in sorted(candidates, key=lambda row: row["reward"], reverse=True)
        ]

    def update_output(self) -> None:
        """Publish the DSE recommendation without interrupting execution on output errors."""
        try:
            tr = self.original_test_run
            output = self.runner.experiment_output
            output.update_dse(str(tr.name), tr.param_space, self._ranked_dse_candidates())
            output.write()
        except Exception as exc:
            logging.warning("Cannot update DSE output for %s: %s", self.original_test_run.name, exc)

    @property
    def upcoming_trial(self) -> int:
        """
        Index of the next trial ``step`` will run.

        Logical progress includes executed, cached, failed, and skipped steps. ``TestRun.step`` is
        assigned this index only for accepted candidates, so rejected proposals below the retry
        limit do not consume a trial number. ``reset`` peeks at the same next value to report the
        regime the first ``step`` will apply.
        """
        return self.progress + 1

    @property
    def progress(self) -> int:
        """Number of logical sweep steps consumed, including skipped steps."""
        return self._progress

    def _advance_progress(self) -> None:
        self._progress += 1
        self._consecutive_rejections = 0

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
        info: dict[str, Any] = {}
        if self.params is not None:
            info["env_params"] = self.params.sample(self.upcoming_trial)
        return self.define_observation_space(), info

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        trial = self.upcoming_trial
        sampled_env_params = self.params.sample(trial) if self.params else {}
        info: dict[str, Any] = {"env_params": sampled_env_params} if self.params is not None else {}
        candidate_test_run = self.test_run.apply_params_set(action, env_params=sampled_env_params)
        candidate_test_run.step = trial
        cached_result = self.get_cached_trajectory_result(action, sampled_env_params, test_run=candidate_test_run)
        failed_metrics: list[str] = []
        execution_error = None

        if cached_result is not None:
            self.test_run = candidate_test_run
            self.test_run.output_path = self.iteration_dir / str(trial)
            logging.info(
                "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                cached_result["reward"],
                cached_result["step"],
            )
            if "observation" in cached_result:
                observation = dict(zip(self.test_run.test.agent_metrics, cached_result["observation"], strict=True))
            else:
                observation = {
                    metric: cached_result[f"observation.{metric}"] for metric in self.test_run.test.agent_metrics
                }
            reward = cast(float, cached_result["reward"])
        else:
            if not candidate_test_run.test.constraint_check(candidate_test_run, self.runner.system):
                logging.info("Constraint check failed for step %s", trial)
                self._consecutive_rejections += 1
                if self._consecutive_rejections >= self.max_consecutive_rejected_proposals:
                    logging.info("Skipping DSE step %s after repeated constraint rejections", trial)
                    self._advance_progress()
                return [-1.0], self.rewards.constraint_failure, True, info

            submitted_test_run = copy.deepcopy(candidate_test_run)
            submitted_test_run.output_path = self.runner.get_job_output_path(submitted_test_run)
            self.test_run = submitted_test_run
            self.runner.test_scenario.test_runs = [submitted_test_run]
            self.runner.shutting_down = False
            self.runner.jobs.clear()
            self.runner.testrun_to_job_map.clear()
            try:
                self.runner.run()
            except Exception as error:
                execution_error = f"{type(error).__name__}: {error}"
                logging.error("Error running step %s: %s", self.test_run.step, error)
            observation = self.get_observation(failed_metrics=failed_metrics)
            reward = self.compute_reward(observation)

        output_test = next(
            (test for test in self.runner.experiment_output.experiment.tests if test.id == self.test_run.name), None
        )
        run_output = (
            next(
                (
                    run
                    for run in reversed(output_test.runs)
                    if Path(run.path).absolute() == self.test_run.output_path.absolute()
                    and run.iteration == self.test_run.current_iteration
                    and run.step == self.test_run.step
                ),
                None,
            )
            if output_test is not None and cached_result is None
            else None
        )
        record = self.trajectory.record(
            test_run=self.test_run,
            action_space=self.original_test_run.param_space,
            action=action,
            reward=reward,
            observation=observation,
            system=self.runner.system,
            rewards=self.rewards,
            failed_metrics=failed_metrics,
            execution_error=execution_error,
            cached_result=cached_result,
            run_output=run_output,
        )
        context = {name: record.get(name, decode_numbers=True) for name in record.metadata}
        info["trajectory"] = context
        self._advance_progress()
        logging.info(
            "Step %s: Observation: %s, Reward: %.4f", trial, [round(obs, 4) for obs in observation.values()], reward
        )
        return list(observation.values()), reward, False, info

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

    def compute_reward(self, observation: dict[str, Any]) -> float:
        """
        Compute a reward based on the TestRun result.

        Args:
            observation: Metric values keyed by configured metric name.

        Returns:
            float: Reward value.
        """
        return self.reward_function(list(observation.values()))

    def get_observation(self, *, failed_metrics: list[str] | None = None) -> dict[str, Any]:
        all_metrics = self.test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")
        observation: dict[str, Any] = {}
        for metric in all_metrics:
            value = self.test_run.get_metric_value(self.runner.system, metric)
            if value is METRIC_ERROR:
                if failed_metrics is not None:
                    failed_metrics.append(metric)
                value = self.rewards.metric_failure
            observation[metric] = value
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

    @property
    def iteration_dir(self) -> Path:
        """Per-iteration output directory containing the trajectory output."""
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}"

    @property
    def trajectory_file_path(self) -> Path:
        return self.trajectory.output_path

    def get_cached_trajectory_result(
        self, action: Any, env_params: dict[str, Any], *, test_run: TestRun | None = None
    ) -> "pd.Series | None":
        candidate = self.test_run.apply_params_set(action, env_params=env_params) if test_run is None else test_run
        return self.trajectory.find(
            candidate, self.original_test_run.param_space, self.runner.system, self.rewards, action=action
        )
