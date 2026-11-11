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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, JobStatusResult, Registry, TestRun
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
    """Independent trial snapshot with ``step`` and ``output_path`` assigned; ``None`` when rejected."""
    cached: "pd.Series | None" = None
    """Trajectory row that already answers this proposal."""

    @property
    def needs_execution(self) -> bool:
        return self.test_run is not None and self.cached is None


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
        """Return valid trial steps and configurations in descending reward order."""
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
        Evaluate proposals in waves of at most ``max_parallelism`` trials.

        A wave is filled from ``proposals``, executed by ``BaseRunner.run_trials``, and completely
        finalized before the next one is filled, so results of earlier waves are in the trajectory
        cache for later proposals. Identical proposals inside one wave are executed twice.

        The runner and its scenario are left as they are: trials are passed to the runner
        explicitly, and a runner shutdown is never reset here, so a cancelled sweep stops after
        the wave that was running.
        """
        results: list[EvaluationResult] = []
        pending = iter(proposals)
        while wave := self._next_wave(pending, max_steps):
            job_statuses = self.runner.run_trials([c.test_run for c in wave if c.test_run and c.needs_execution])
            results.extend(self._finalize_wave(wave, job_statuses))
        return results

    def _next_wave(self, proposals: Iterator[Proposal], max_steps: Optional[int]) -> list[_Candidate]:
        """
        Pull proposals, in order, until ``max_parallelism`` of them need execution.

        Every pulled proposal is returned, so the caller reports one result per proposal:

        - The trial number is the next free one, counting trials accepted earlier in this wave.
          Environment parameters are sampled for that number and are part of the cache key.
        - Cached: a trajectory row matches the action and environment parameters. Consumes a step
          and a trial number, takes no execution slot.
        - Rejected: ``constraint_check`` fails. Consumes nothing, unless it is the
          ``max_consecutive_rejected_proposals``-th rejection in a row, which consumes one step.
          Any accepted proposal resets that counter.
        - Otherwise the proposal takes an execution slot.

        Pulling stops when the wave is full, ``proposals`` ends, or steps consumed so far plus the
        steps this wave will consume reach ``max_steps``. Nothing is pulled once the allowance is
        exhausted or the runner is shutting down, so an empty list ends the sweep.
        """
        del proposals, max_steps
        raise NotImplementedError

    def _finalize_wave(self, wave: list[_Candidate], job_statuses: list[JobStatusResult]) -> list[EvaluationResult]:
        """
        Turn a finished wave into results and persist it, in proposal order.

        ``job_statuses`` is what ``BaseRunner.run_trials`` returned for the executed candidates of
        ``wave``, in the same order. Per candidate:

        - Executed: read metrics from its own ``test_run``; a failed job or missing metric yields
          ``rewards.metric_failure`` and status ``failed``.
        - Cached: reuse the reward and observation of the matching trajectory row.
        - Rejected: ``rewards.constraint_failure``, observation ``[-1.0]``, ``done=True``.

        Executed and cached candidates are appended to the trajectory in trial order, whatever
        order the jobs completed in; rejected ones get no row. ``self.test_run`` ends up as the
        last accepted trial, which is what serial callers read today.
        """
        del wave, job_statuses
        raise NotImplementedError

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
