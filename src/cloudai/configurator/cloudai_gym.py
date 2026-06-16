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
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util import flatten_dict
from cloudai.util.lazy_imports import lazy

from .base_agent import RewardOverrides
from .base_gym import BaseGym
from .env_params import EnvParams, write_env_params


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    env_params: dict[str, Any] = dataclasses.field(default_factory=dict)


class GymServer(Protocol):
    """
    In-process environment server backing ``CloudAIGymEnv``'s online (live-RL) mode.

    An online env delegates each trial to a ``GymServer`` instead of launching a
    workload through the runner: the server owns the simulation/state and returns
    the reward directly, so the agent-facing interface is identical to the
    runner-backed mode. Online mode is selected when ``cmd_args.live_rl_mode`` is
    true (the server is built from ``cmd_args.env_class``) or when a server is
    passed explicitly to :class:`CloudAIGymEnv`.
    """

    def reset(self) -> Tuple[List[float], Dict[str, Any]]: ...

    def step(self, action: Dict[str, Any]) -> Tuple[List[float], float, bool, Dict[str, Any]]: ...

    def get_action_space(self) -> Dict[str, Any]: ...

    def get_observation_space(self) -> List[float]: ...


def _create_gym_server(test_run: TestRun) -> GymServer:
    """
    Instantiate the ``GymServer`` named by ``cmd_args.env_class`` for online mode.

    ``env_class`` is a dotted import path (``"pkg.module.Class"``). The remaining
    cmd_args are passed as keyword arguments, filtered to the server's
    ``__init__`` signature unless it accepts ``**kwargs``; framework-only keys
    (``live_rl_mode``, ``docker_image_url``) are dropped.
    """
    args_dict = flatten_dict(test_run.test.cmd_args.model_dump())

    env_class_path = args_dict.pop("env_class", None)
    if not env_class_path:
        raise ValueError(
            "online mode (live_rl_mode=true) requires 'env_class' in cmd_args pointing to a GymServer class"
        )
    for key in ("live_rl_mode", "docker_image_url"):
        args_dict.pop(key, None)

    module_path, class_name = str(env_class_path).rsplit(".", 1)
    server_cls = getattr(importlib.import_module(module_path), class_name)

    sig = inspect.signature(server_cls.__init__)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_kwargs:
        valid = set(sig.parameters) - {"self"}
        args_dict = {k: v for k, v in args_dict.items() if k in valid}

    return server_cls(**args_dict)


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
        gym_server: GymServer | None = None,
    ):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
            gym_server: Optional in-process server enabling online (live-RL) mode. When omitted,
                online mode is auto-detected from ``cmd_args.live_rl_mode``; otherwise the env runs
                in the default runner-backed mode.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.trajectory: dict[int, list[TrajectoryEntry]] = {}
        self.params: EnvParams | None = EnvParams.from_test(test_run.test)
        self._online_step_count = 0
        if gym_server is not None:
            self._gym_server: GymServer | None = gym_server
        elif getattr(test_run.test.cmd_args, "live_rl_mode", False):
            self._gym_server = _create_gym_server(test_run)
        else:
            self._gym_server = None
        super().__init__()

    @property
    def _is_online(self) -> bool:
        """True when the env delegates steps to an in-process GymServer (live-RL mode)."""
        return self._gym_server is not None

    @property
    def env_params_record_path(self) -> Path:
        """``env.csv`` lives alongside ``trajectory.csv`` so a plain ``merge`` joins them."""
        return self.iteration_dir / "env.csv"

    def define_action_space(self) -> Dict[str, Any]:
        server = self._gym_server
        if server is not None:
            return server.get_action_space()
        return self.test_run.param_space

    @property
    def first_sweep(self) -> dict[str, Any]:
        """Builds a sweep using first elements of each explorable parameter."""
        return {k: v[0] for k, v in self.define_action_space().items()}

    def define_observation_space(self) -> list:
        """
        Define the observation space for the environment.

        Returns:
            list: In online mode, the GymServer's observation space. Otherwise one float slot per
            agent metric (at least one), giving the correct shape for adapters that derive
            ``gymnasium.spaces.Box`` from this output.
        """
        server = self._gym_server
        if server is not None:
            return server.get_observation_space()
        return [0.0] * max(len(self.test_run.test.agent_metrics), 1)

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
        server = self._gym_server
        if server is not None:
            self._online_step_count = 0
            return server.reset()
        self.test_run.current_iteration = 0
        observation = self.define_observation_space()
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
        server = self._gym_server
        if server is not None:
            return self._online_step(server, action)

        self.test_run.increment_step()
        # RNG lives in the env: sample here, then apply action + sample so the run and cache key see them.
        sampled_env_params = self.params.sample(self.test_run.step) if self.params else {}
        self.test_run = self.test_run.apply_params_set(action, env_params=sampled_env_params)

        cached_result = self.get_cached_trajectory_result(action, sampled_env_params)
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
                    env_params=sampled_env_params,
                )
            )
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

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)

        self.write_trajectory(
            TrajectoryEntry(
                step=self.test_run.step,
                action=action,
                reward=reward,
                observation=observation,
                env_params=sampled_env_params,
            )
        )

        return observation, reward, False, {}

    def _online_step(self, server: GymServer, action: Any) -> Tuple[list, float, bool, dict]:
        """
        Execute one step against the in-process GymServer (online / live-RL mode).

        Bypasses the runner, constraint check, env_params observers, and trajectory cache: the
        server owns the simulation and returns the reward directly. A trajectory row is still
        written so online runs produce the same ``trajectory.csv`` artifact.
        """
        self._online_step_count += 1
        observation, reward, done, info = server.step(action)
        self.write_trajectory(
            TrajectoryEntry(
                step=self._online_step_count,
                action=action,
                reward=reward,
                observation=observation,
            )
        )
        return observation, reward, done, info

    def render(self, mode: str = "human"):
        """
        Render the current state of the TestRun.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        if self._is_online:
            logging.info(f"CloudAIGymEnv [online] step {self._online_step_count}")
        else:
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

        write_env_params(self.env_params_record_path, entry.step, entry.env_params)

    @property
    def iteration_dir(self) -> Path:
        """Per-iteration output dir; trajectory.csv and env.csv both live here, step-aligned."""
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}"

    @property
    def trajectory_file_path(self) -> Path:
        if self._is_online:
            return self.test_run.output_path / "trajectory.csv"
        return self.iteration_dir / "trajectory.csv"

    @property
    def current_trajectory(self) -> list[TrajectoryEntry]:
        return self.trajectory.setdefault(self.test_run.current_iteration, [])

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
        for entry in self.current_trajectory:
            action_match = self._values_match_exact(entry.action, action)
            env_params_match = self._values_match_exact(entry.env_params, env_params)
            if action_match and env_params_match:
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
