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
import math
import pathlib
import time
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from cloudai import core
from cloudai.configurator import base_agent, base_gym, trajectory
from cloudai.configurator import env_params as env_params_lib
from cloudai.util import lazy_imports

if TYPE_CHECKING:
    import pandas as pd


class CloudAIGymEnv(base_gym.BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(
        self,
        test_run: core.TestRun,
        runner: core.BaseRunner,
        rewards: base_agent.RewardOverrides,
        max_consecutive_rejected_proposals: int = base_agent.BaseAgentConfig().max_consecutive_rejected_proposals,
        max_parallelism: int = 1,
    ):
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.max_consecutive_rejected_proposals = max_consecutive_rejected_proposals
        self.max_parallelism = max_parallelism
        self._progress = test_run.step
        self._consecutive_rejections = 0
        self.reward_function = core.Registry().get_reward_function(test_run.test.agent_reward_function)
        self.params: env_params_lib.EnvParams | None = env_params_lib.EnvParams.from_test(test_run.test)
        self.trajectory = trajectory.Trajectory(iteration_dir=self.iteration_dir)
        super().__init__()

    def _ranked_dse_candidates(self) -> list[tuple[int, dict[str, str | int | float]]]:
        tr = self.original_test_run
        candidates = [
            row
            for row in self.trajectory.dataframe.to_dict(orient="records")
            if math.isfinite(row["reward"])
            and all(
                math.isfinite(row[f"observation.{metric}"])
                and row[f"observation.{metric}"] != self.rewards.metric_failure
                for metric in tr.test.agent_metrics
            )
        ]
        return [
            (int(row["step"]), {key: row[f"action.{key}"] for key in tr.param_space})
            for row in sorted(candidates, key=lambda row: (-row["reward"], row["step"]))
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
        """Retries reuse a trial number so environment sampling stays stable."""
        return self._progress + 1

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
            lazy_imports.lazy.np.random.seed(seed)
        info: dict[str, Any] = {}
        if self.params is not None:
            info["env_params"] = self.params.sample(self.upcoming_trial)
        return self.define_observation_space(), info

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        results = self.evaluate_batch([(self.upcoming_trial, action)])
        if not results:
            raise RuntimeError("Cannot evaluate the action: the runner is shutting down.")
        result = results[0]
        info: dict[str, Any] = {"env_params": result.env_params} if self.params is not None else {}
        return result.observation, result.reward, result.done, info

    def evaluate_batch(
        self, proposals: Iterable[base_gym.Proposal], max_steps: Optional[int] = None
    ) -> list[base_gym.Trial]:
        """Preserve proposal order for agent feedback despite out-of-order completion."""
        pending = iter(proposals)
        trials: list[base_gym.Trial] = []
        executing: dict[int, base_gym.Trial] = {}
        followers: dict[int, list[base_gym.Trial]] = {}
        try:
            self._start_trials(pending, max_steps, trials, executing, followers)
            while self.runner.jobs:
                self._poll_trials(executing, followers)
                still_running = bool(self.runner.jobs)
                self._start_trials(pending, max_steps, trials, executing, followers)
                if still_running:
                    time.sleep(self.runner.monitor_interval)
        except BaseException:
            self.runner.shutdown()
            raise
        unfinished = sum(trial.status == "pending" for trial in trials)
        if self.runner.shutting_down and unfinished:
            logging.info("DSE is cancelled; %s unfinished proposal(s) are not recorded.", unfinished)
        return [trial for trial in trials if trial.status != "pending"]

    def _start_trials(
        self,
        proposals: Iterator[base_gym.Proposal],
        max_steps: int | None,
        trials: list[base_gym.Trial],
        executing: dict[int, base_gym.Trial],
        followers: dict[int, list[base_gym.Trial]],
    ) -> None:
        while len(self.runner.jobs) < self.max_parallelism and not self.runner.shutting_down:
            if max_steps is not None and self._progress >= max_steps:
                break
            proposal = next(proposals, None)
            if proposal is None:
                break

            trial_index, action = proposal
            action = copy.deepcopy(action)  # the agent may reuse its dict for the next proposal
            step = self.upcoming_trial
            env_params = self.params.sample(step) if self.params else {}
            test_run = self.test_run.apply_params_set(action, env_params=env_params)
            test_run.step = step
            trial = base_gym.Trial(trial_index, action, env_params=env_params, step=step)
            trials.append(trial)
            logging.info("Proposal %s for step %s: %s", trial_index, step, action)

            cached = self.get_cached_trajectory_result(action, env_params)
            owner = next(
                (
                    number
                    for number, running in executing.items()
                    if trajectory.Trajectory.values_match(running.action, action)
                    and trajectory.Trajectory.values_match(running.env_params, env_params)
                ),
                None,
            )
            if cached is not None:
                logging.info(
                    "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                    cached["reward"],
                    cached["step"],
                )
                trial.status = "cached"
                trial.observation = [cached[f"observation.{metric}"] for metric in test_run.test.agent_metrics]
                trial.reward = float(cached["reward"])
                trial.output_path = self.iteration_dir / str(cached["step"])
                self._record(trial, test_run)
            elif owner is not None:
                logging.info("Step %s shares the result of step %s, which is still running.", step, owner)
                followers.setdefault(owner, []).append(trial)
            elif not test_run.test.constraint_check(test_run, self.runner.system):
                logging.info("Constraint check failed for step %s", step)
                trial.step = None
                trial.status = "rejected"
                trial.observation = [-1.0]
                trial.reward = self.rewards.constraint_failure
                self._record(trial)
            else:
                test_run.output_path = self.runner.get_job_output_path(test_run)
                self._submit_trial(trial, test_run, executing)

            if trial.step is None:
                self._consecutive_rejections += 1
                if self._consecutive_rejections < self.max_consecutive_rejected_proposals:
                    continue
                logging.info("Skipping DSE step %s after repeated constraint rejections", step)
            self._progress += 1
            self._consecutive_rejections = 0

    def _submit_trial(
        self, trial: base_gym.Trial, test_run: core.TestRun, executing: dict[int, base_gym.Trial]
    ) -> None:
        error = "the trial was not started"  # e.g. the runner is shutting down
        try:
            self.runner.submit_test(test_run)
        except core.JobSubmissionError:
            raise
        except Exception as e:
            logging.error(f"Error running step {test_run.step}: {e}")
            error = str(e)
        if any(job.test_run is test_run for job in self.runner.jobs):
            executing[test_run.step] = trial
            logging.info(f"Running step {test_run.step}; {len(self.runner.jobs)} trial(s) in progress")
        elif not self.runner.shutting_down:
            self._complete_trial(trial, test_run, core.JobStatusResult(is_successful=False, error_message=error))
            self._record(trial, test_run)

    def _poll_trials(self, executing: dict[int, base_gym.Trial], followers: dict[int, list[base_gym.Trial]]) -> None:
        """Bypass scenario fail-fast handling so trial failures stay isolated."""
        completed = list(self.runner.poll_jobs())
        ready: list[tuple[base_gym.Trial, core.TestRun, core.JobStatusResult]] = []
        for job, job_status in completed:
            trial = executing[job.test_run.step]
            if not job_status.is_successful:
                if self.runner.shutting_down:
                    logging.info("Step %s stopped during shutdown.", job.test_run.step)
                else:
                    logging.error(f"Step {job.test_run.step} failed: {job_status.error_message}")
            if self.runner.shutting_down:
                continue
            ready.append((trial, job.test_run, job_status))
            for follower in followers.pop(job.test_run.step, []):
                ready.append((follower, job.test_run, job_status))

        # Keep proposal order among outcomes discovered by the same polling pass, recording
        # each outcome before reading the next one's metrics so an error preserves prior rows.
        for trial, test_run, job_status in sorted(ready, key=lambda item: item[0].step or 0):
            if self.runner.shutting_down:
                break
            if trial.step == test_run.step:
                self._complete_trial(trial, test_run, job_status)
            else:
                source = executing[test_run.step]
                trial.status = "cached"
                trial.observation = list(source.observation)
                trial.reward = source.reward
                trial.output_path = source.output_path
            self._record(trial, test_run)
        for job, _ in completed:
            del executing[job.test_run.step]

    def _complete_trial(self, trial: base_gym.Trial, test_run: core.TestRun, job_status: core.JobStatusResult) -> None:
        """Read metrics from the completed run, not the serial caller's current run."""
        observation = self.get_observation(test_run)
        trial.observation = list(observation.values())
        trial.reward = self.compute_reward(observation)
        failed_metrics = [key for key, value in observation.items() if value == self.rewards.metric_failure]
        if failed_metrics:
            logging.warning(
                "Step %s has missing or failed metrics %s in %s.", trial.step, failed_metrics, test_run.output_path
            )
        trial.status = "failed" if failed_metrics or not job_status.is_successful else "completed"
        trial.output_path = test_run.output_path

    def _record(self, trial: base_gym.Trial, test_run: core.TestRun | None = None) -> None:
        """Update the serial view for callers that inspect env.test_run after stepping."""
        if self.runner.shutting_down:
            trial.status = "pending"
            return
        if trial.done:
            return
        if test_run is None or trial.step is None or trial.output_path is None:
            raise ValueError("Recording an accepted trial requires its run, step and output path")
        if test_run.step != trial.step:
            test_run = copy.deepcopy(test_run)
            test_run.step = trial.step
        test_run.output_path = trial.output_path
        self.test_run = test_run
        observation = dict(zip(test_run.test.agent_metrics, trial.observation, strict=True))

        self.trajectory.append(
            step=trial.step,
            action=trial.action,
            reward=trial.reward,
            observation=observation,
            env_params=trial.env_params,
        )
        logging.info(
            "Step %s: Observation: %s, Reward: %.4f",
            trial.step,
            [round(obs, 4) for obs in trial.observation],
            trial.reward,
        )

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
            lazy_imports.lazy.np.random.seed(seed)

    def compute_reward(self, observation: dict[str, Any]) -> float:
        """
        Compute a reward based on the TestRun result.

        Args:
            observation: Metric values keyed by configured metric name.

        Returns:
            float: Reward value.
        """
        return self.reward_function(list(observation.values()))

    def get_observation(self, test_run: core.TestRun | None = None) -> dict[str, Any]:
        test_run = self.test_run if test_run is None else test_run
        all_metrics = test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")

        observation: dict[str, Any] = {}
        for metric in all_metrics:
            v = test_run.get_metric_value(self.runner.system, metric)
            if v is core.METRIC_ERROR:
                v = self.rewards.metric_failure
            observation[metric] = v
        return observation

    def structured_observation_descriptors(self) -> Optional[Dict[str, env_params_lib.ObsLeafDescriptor]]:
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
    def iteration_dir(self) -> pathlib.Path:
        """Per-iteration output directory containing the trajectory output."""
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}"

    @property
    def trajectory_file_path(self) -> pathlib.Path:
        return self.trajectory.output_path

    def get_cached_trajectory_result(self, action: Any, env_params: dict[str, Any]) -> "pd.Series | None":
        """Return the first trajectory row matching the action and environment parameters."""
        return self.trajectory.find(action=action, env_params=env_params)
