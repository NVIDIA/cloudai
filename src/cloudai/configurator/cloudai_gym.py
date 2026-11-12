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
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, JobStatusResult, JobSubmissionError, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_agent import BaseAgentConfig, RewardOverrides
from .base_gym import BaseGym, EvaluationResult, Proposal
from .env_params import EnvParams, ObsLeafDescriptor
from .trajectory import Trajectory

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class _Candidate:
    """One consumed proposal on its way to an ``EvaluationResult``."""

    trial_index: int
    action: dict[str, Any]
    env_params: dict[str, Any]
    test_run: TestRun | None = None
    """Independent trial snapshot with ``step`` assigned; ``None`` when rejected."""
    cached: "pd.Series | None" = None
    """Trajectory row that already answers this proposal."""
    duplicate_of: "_Candidate | None" = None
    """Earlier candidate, not recorded yet, that executes the same proposal."""
    job_status: JobStatusResult | None = None
    """Set once the trial is over; only for candidates that need execution."""
    observation: dict[str, Any] | None = None
    reward: float = 0.0
    result: EvaluationResult | None = None
    """Set once the candidate is recorded."""

    @property
    def step(self) -> int:
        """Trial number; ``0`` when the proposal was rejected."""
        return self.test_run.step if self.test_run is not None else 0

    @property
    def needs_execution(self) -> bool:
        return self.test_run is not None and self.cached is None and self.duplicate_of is None

    @property
    def is_finished(self) -> bool:
        """Whether the candidate can be recorded."""
        if self.duplicate_of is not None:
            return self.duplicate_of.result is not None
        return not self.needs_execution or self.job_status is not None

    def is_same_proposal(self, action: dict[str, Any], env_params: dict[str, Any]) -> bool:
        return Trajectory.values_match(self.action, action) and Trajectory.values_match(self.env_params, env_params)


@dataclass
class _Batch:
    """State of one ``evaluate_batch`` call."""

    proposals: Iterator[Proposal]
    max_steps: Optional[int]
    candidates: list[_Candidate] = field(default_factory=list)
    """Every pulled proposal, in proposal order."""
    waiting: list[_Candidate] = field(default_factory=list)
    """The candidates that are not recorded yet, in proposal order."""


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
        max_parallelism: int = 1,
    ):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
            max_consecutive_rejected_proposals: Rejections allowed before consuming the current sweep step.
            max_parallelism: Maximum number of trials kept outstanding at the same time.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.max_consecutive_rejected_proposals = max_consecutive_rejected_proposals
        self.max_parallelism = max_parallelism
        self._progress = test_run.step
        self._consecutive_rejections = 0
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.params: EnvParams | None = EnvParams.from_test(test_run.test)
        self.trajectory = Trajectory(iteration_dir=self.iteration_dir)
        super().__init__()

    def _ranked_dse_candidates(self) -> list[tuple[int, dict[str, str | int | float]]]:
        """Return valid trial steps and configurations in descending reward order, equal rewards by step."""
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
        """
        Index of the next trial the environment will assign.

        Executed, cached, failed, and skipped steps consume an index. ``TestRun.step`` is assigned
        this index only for accepted proposals, so rejected proposals below the retry limit do not
        consume a trial number. ``reset`` peeks at the same next value to report the regime the
        next accepted proposal will get.
        """
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
            lazy.np.random.seed(seed)
        self.test_run.current_iteration = 0
        info: dict[str, Any] = {}
        if self.params is not None:
            info["env_params"] = self.params.sample(self.upcoming_trial)
        return self.define_observation_space(), info

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        """
        Execute one step in the environment: a one-proposal ``evaluate_batch``.

        Args:
            action (Any): Action chosen by the agent.

        Returns:
            Tuple: A tuple containing:
                - observation (list): Updated system state.
                - reward (float): Reward for the action taken.
                - done (bool): ``True`` when the action was rejected by the constraint check.
                - info (dict): Additional info for debugging.
        """
        results = self.evaluate_batch([(self.upcoming_trial, action)])
        if not results:
            raise RuntimeError("Cannot evaluate the action: the runner is shutting down.")
        result = results[0]
        info: dict[str, Any] = {"env_params": result["env_params"]} if self.params is not None else {}
        return result["observation"], result["value"], result["done"], info

    def evaluate_batch(self, proposals: Iterable[Proposal], max_steps: Optional[int] = None) -> list[EvaluationResult]:
        """
        Evaluate proposals, keeping up to ``max_parallelism`` trials running.

        A proposal is pulled from ``proposals`` as soon as a slot is free, so a slow trial holds
        its own slot and not the others. Trials are recorded as they finish: the trajectory is in
        completion order, the returned results are in proposal order. A proposal identical to a
        trial that is still running shares that trial's result.

        The runner's scenario is left as it is and a runner shutdown is never reset here: a
        cancelled sweep stops, and the proposals that were not recorded by then get no result.

        A trial that cannot be prepared, and an error while checking the running trials, fail the
        affected trials and the sweep goes on. Any other error (from the agent, from recording, a
        job submission error, an interrupt) terminates the running trials and is raised.
        """
        batch = _Batch(iter(proposals), max_steps)
        try:
            self._start_trials(batch)
            while self.runner.jobs:
                self._poll_trials(batch)
                still_running = bool(self.runner.jobs)
                self._start_trials(batch)  # refill the free slots before waiting
                if still_running:
                    time.sleep(self.runner.monitor_interval)
        except BaseException:
            self.runner.abandon_jobs()
            raise
        if self.runner.shutting_down and batch.waiting:
            logging.info("DSE is cancelled; %s unfinished proposal(s) are not recorded.", len(batch.waiting))
        return [candidate.result for candidate in batch.candidates if candidate.result is not None]

    def _start_trials(self, batch: _Batch) -> None:
        """
        Pull proposals, in order, until ``max_parallelism`` trials are running.

        For every pulled proposal:

        - The trial number is the next free one. Environment parameters are sampled for that
          number and are part of the cache key.
        - Cached: a trajectory row, or a trial that is still running, matches the action and
          environment parameters. Consumes a step and a trial number, takes no execution slot.
        - Rejected: ``constraint_check`` fails. Consumes nothing, unless it is the
          ``max_consecutive_rejected_proposals``-th rejection in a row, which consumes one step.
          Any accepted proposal resets that counter.
        - Otherwise the trial is submitted and takes an execution slot. A trial that cannot be
          prepared for submission fails alone; a job submission error is raised.

        Pulling stops when every slot is taken, ``proposals`` ends, ``max_steps`` steps are
        consumed, or the runner is shutting down.
        """
        while len(self.runner.jobs) < self.max_parallelism and not self.runner.shutting_down:
            if batch.max_steps is not None and self._progress >= batch.max_steps:
                break
            proposal = next(batch.proposals, None)
            if proposal is None:
                break

            trial_index, action = proposal
            action = copy.deepcopy(action)  # the agent may reuse its dict for the next proposal
            trial = self.upcoming_trial
            env_params = self.params.sample(trial) if self.params else {}
            test_run = self.test_run.apply_params_set(action, env_params=env_params)
            test_run.step = trial
            logging.info("Proposal %s for step %s: %s", trial_index, trial, action)

            cached = self.get_cached_trajectory_result(action, env_params)
            running = next(
                (c for c in batch.waiting if c.needs_execution and c.is_same_proposal(action, env_params)), None
            )
            if cached is not None:
                logging.info(
                    "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                    cached["reward"],
                    cached["step"],
                )
                candidate = _Candidate(trial_index, action, env_params, test_run, cached)
            elif running is not None:
                logging.info("Step %s shares the result of step %s, which is still running.", trial, running.step)
                candidate = _Candidate(trial_index, action, env_params, test_run, duplicate_of=running)
            elif not test_run.test.constraint_check(test_run, self.runner.system):
                logging.info("Constraint check failed for step %s", trial)
                candidate = _Candidate(trial_index, action, env_params)
            else:
                test_run.output_path = self.runner.get_job_output_path(test_run)
                candidate = _Candidate(trial_index, action, env_params, test_run)
                self._submit_trial(candidate, test_run)
            batch.candidates.append(candidate)
            batch.waiting.append(candidate)
            self._record_finished(batch)  # so that the next proposal finds it in the trajectory

            if candidate.test_run is None:
                self._consecutive_rejections += 1
                if self._consecutive_rejections < self.max_consecutive_rejected_proposals:
                    continue
                logging.info("Skipping DSE step %s after repeated constraint rejections", trial)
            self._progress += 1
            self._consecutive_rejections = 0

    def _submit_trial(self, candidate: _Candidate, test_run: TestRun) -> None:
        error = "the trial was not started"  # e.g. the runner is shutting down
        try:
            self.runner.submit_test(test_run)
        except JobSubmissionError:
            raise
        except Exception as e:
            logging.error(f"Error running step {candidate.step}: {e}")
            error = str(e)
        if not any(job.test_run is test_run for job in self.runner.jobs):
            # There is no job to wait for, so the trial is over.
            candidate.job_status = JobStatusResult(is_successful=False, error_message=error)
            return
        logging.info(f"Running step {candidate.step}; {len(self.runner.jobs)} trial(s) in progress")

    def _poll_trials(self, batch: _Batch) -> None:
        """
        Check the running trials once and record the ones that finished.

        Uses the runner's own polling without its scenario policy: a failed trial does not stop
        the others. An error while checking fails the trials that are still running, whose jobs
        are terminated, and the sweep goes on.
        """
        running = {candidate.step: candidate for candidate in batch.waiting if candidate.needs_execution}
        try:
            for job, job_status in self.runner.poll_jobs():
                running[job.test_run.step].job_status = job_status
                if not job_status.is_successful:
                    logging.error(f"Step {job.test_run.step} failed: {job_status.error_message}")
        except JobSubmissionError:
            raise
        except Exception as e:
            steps = sorted(job.test_run.step for job in self.runner.jobs)
            self.runner.abandon_jobs()
            logging.error(f"Error while waiting for steps {steps}: {e}")
            for step in steps:
                running[step].job_status = JobStatusResult(is_successful=False, error_message=str(e))
        self._record_finished(batch)

    def _record_finished(self, batch: _Batch) -> None:
        """
        Record every waiting candidate that is finished, in proposal order.

        Nothing is recorded once the runner is shutting down: a terminated trial cannot be told
        from a failed one.
        """
        if self.runner.shutting_down:
            return
        waiting: list[_Candidate] = []
        for candidate in batch.waiting:
            if candidate.is_finished:
                candidate.result = self._record(candidate)
            else:
                waiting.append(candidate)
        batch.waiting = waiting

    def _record(self, candidate: _Candidate) -> EvaluationResult:
        """
        Turn a finished candidate into its result and persist it.

        - Executed: read metrics from its own ``test_run``. Missing metrics are reported as
          ``rewards.metric_failure``; such a metric or a failed job makes the status ``failed``.
        - Cached: reuse the reward and observation of the matching trajectory row or of the
          trial it duplicates.
        - Rejected: ``rewards.constraint_failure``, observation ``[-1.0]``, ``done=True``.

        Executed and cached candidates get a trajectory row; rejected ones do not.
        ``self.test_run`` becomes the recorded trial, which is what serial callers read; for a
        cached one its ``output_path`` is the directory of the trial whose result is reused.
        """
        if candidate.test_run is None:
            return EvaluationResult(
                trial_index=candidate.trial_index,
                action=candidate.action,
                value=self.rewards.constraint_failure,
                observation=[-1.0],
                done=True,
                status="rejected",
                step=None,
                output_path=None,
                env_params=candidate.env_params,
            )

        self.test_run = candidate.test_run
        status: Literal["completed", "failed", "cached"] = "cached"
        if candidate.cached is not None:
            metrics = self.test_run.test.agent_metrics
            candidate.observation = {metric: candidate.cached[f"observation.{metric}"] for metric in metrics}
            candidate.reward = float(candidate.cached["reward"])
            output_path = self.test_run.output_path = self.iteration_dir / str(candidate.cached["step"])
        elif candidate.duplicate_of is not None and candidate.duplicate_of.test_run is not None:
            candidate.observation = dict(candidate.duplicate_of.observation or {})
            candidate.reward = candidate.duplicate_of.reward
            output_path = self.test_run.output_path = candidate.duplicate_of.test_run.output_path
        else:
            candidate.observation = self.get_observation()
            candidate.reward = self.compute_reward(candidate.observation)
            has_metric_failure = any(value == self.rewards.metric_failure for value in candidate.observation.values())
            job_failed = candidate.job_status is None or not candidate.job_status.is_successful
            status = "failed" if has_metric_failure or job_failed else "completed"
            output_path = self.test_run.output_path

        self.trajectory.append(
            step=candidate.step,
            action=candidate.action,
            reward=candidate.reward,
            observation=candidate.observation,
            env_params=candidate.env_params,
        )
        logging.info(
            "Step %s: Observation: %s, Reward: %.4f",
            candidate.step,
            [round(obs, 4) for obs in candidate.observation.values()],
            candidate.reward,
        )
        return EvaluationResult(
            trial_index=candidate.trial_index,
            action=candidate.action,
            value=candidate.reward,
            observation=list(candidate.observation.values()),
            done=False,
            status=status,
            step=candidate.step,
            output_path=output_path,
            env_params=candidate.env_params,
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

    def get_observation(self) -> dict[str, Any]:
        """
        Get the observation from the TestRun object.

        Returns:
            Metric values keyed by configured metric name.
        """
        all_metrics = self.test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")

        observation: dict[str, Any] = {}
        for metric in all_metrics:
            v = self.test_run.get_metric_value(self.runner.system, metric)
            if v is METRIC_ERROR:
                v = self.rewards.metric_failure
            observation[metric] = v
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

    def get_cached_trajectory_result(self, action: Any, env_params: dict[str, Any]) -> "pd.Series | None":
        """Return the first trajectory row matching the action and environment parameters."""
        return self.trajectory.find(action=action, env_params=env_params)
