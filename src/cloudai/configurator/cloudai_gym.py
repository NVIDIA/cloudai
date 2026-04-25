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
import logging
import random as stdlib_random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_gym import BaseGym


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class GymServer(Protocol):
    """Protocol for gym server objects used in online mode."""

    def reset(self) -> Tuple[List[float], Dict[str, Any]]: ...
    def step(self, action: Dict[str, Any]) -> Tuple[List[float], float, bool, Dict[str, Any]]: ...
    def get_action_space(self) -> Dict[str, Any]: ...
    def get_observation_space(self) -> List[float]: ...


class _StepBackend(Protocol):
    """Internal protocol for execution backends."""

    def get_action_space(self) -> Dict[str, Any]: ...
    def get_observation_space(self) -> list: ...
    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]: ...
    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]: ...


class _RunnerBackend:
    """Backend that launches real workloads via the CloudAI runner."""

    def __init__(self, test_run: TestRun, runner: BaseRunner) -> None:
        self._test_run = test_run
        self._original_test_run = copy.deepcopy(test_run)
        self._runner = runner
        self._trajectory_cache: dict[int, list[TrajectoryEntry]] = {}

    @property
    def test_run(self) -> TestRun:
        return self._test_run

    @test_run.setter
    def test_run(self, value: TestRun) -> None:
        self._test_run = value

    def get_action_space(self) -> Dict[str, Any]:
        return self._test_run.param_space

    def get_observation_space(self) -> list:
        n_metrics = max(len(self._test_run.test.agent_metrics), 1)
        return [0.0] * n_metrics

    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]:
        if seed is not None:
            lazy.np.random.seed(seed)
        self._test_run.current_iteration = 0
        return self.get_observation_space(), {}

    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]:
        self._test_run = self._test_run.apply_params_set(action)

        cached = self._get_cached_result(action)
        if cached is not None:
            logging.info("Retrieved cached result with reward %s. Skipping step.", cached.reward)
            return cached.observation, False, cached.info

        if not self._test_run.test.constraint_check(self._test_run, self._runner.system):
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], True, {"reason": "constraint_check_failed"}

        new_tr = copy.deepcopy(self._test_run)
        new_tr.output_path = self._runner.get_job_output_path(new_tr)
        self._runner.test_scenario.test_runs = [new_tr]

        self._runner.shutting_down = False
        self._runner.jobs.clear()
        self._runner.testrun_to_job_map.clear()

        try:
            self._runner.run()
        except Exception as e:
            logging.error(f"Error running step {self._test_run.step}: {e}")

        if self._runner.test_scenario.test_runs and self._runner.test_scenario.test_runs[0].output_path.exists():
            self._test_run = self._runner.test_scenario.test_runs[0]
        else:
            self._test_run = copy.deepcopy(self._original_test_run)
            self._test_run.step = new_tr.step
            self._test_run.output_path = new_tr.output_path

        observation = self._get_observation(action)
        return observation, False, {}

    def get_observation(self, action: Any) -> list:
        return self._get_observation(action)

    def _get_observation(self, action: Any) -> list:
        all_metrics = self._test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")

        observation = []
        for metric in all_metrics:
            v = self._test_run.get_metric_value(self._runner.system, metric)
            if v == METRIC_ERROR:
                v = -1.0
            observation.append(v)
        return observation

    def cache_trajectory(self, entry: TrajectoryEntry) -> None:
        self._trajectory_cache.setdefault(self._test_run.current_iteration, []).append(entry)

    def _get_cached_result(self, action: Any) -> Optional[TrajectoryEntry]:
        for entry in self._trajectory_cache.get(self._test_run.current_iteration, []):
            if _values_match_exact(entry.action, action):
                return entry
        return None


class _GymServerBackend:
    """Backend that delegates to an in-process GymServer for fast, stateful interaction."""

    def __init__(self, server: Any) -> None:
        self._server = server
        self._step_count = 0

    def get_action_space(self) -> Dict[str, Any]:
        return self._server.get_action_space()

    def get_observation_space(self) -> list:
        return self._server.get_observation_space()

    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]:
        self._step_count = 0
        return self._server.reset()

    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]:
        self._step_count += 1
        observation, _raw_reward, done, info = self._server.step(action)
        return observation, done, info


def _create_gym_server(test_run: TestRun) -> Any:
    """Instantiate a GymServer from the env_class path in cmd_args."""
    from cloudai.util import flatten_dict

    cmd_args = test_run.test.cmd_args
    args_dict = flatten_dict(cmd_args.model_dump())

    env_class_path = args_dict.pop("env_class", None)
    if not env_class_path:
        raise ValueError("online mode requires 'env_class' in cmd_args pointing to a GymServer class")

    for key in ("live_rl_mode", "docker_image_url"):
        args_dict.pop(key, None)

    module_path, class_name = env_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    server_cls = getattr(module, class_name)

    import inspect
    sig = inspect.signature(server_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    if valid_params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        args_dict = {k: v for k, v in args_dict.items() if k in valid_params}

    return server_cls(**args_dict)


class CloudAIGymEnv(BaseGym):
    """Unified Gym environment for CloudAI.

    Supports two execution modes selected automatically:

    - **Runner mode** (default): launches real workloads via the CloudAI runner,
      reads metrics from job output.  Used for standard DSE.
    - **Online mode** (``live_rl_mode=true`` in cmd_args): delegates to an
      in-process GymServer for fast, stateful interaction.  Used for
      online RL / simulation-based optimization (e.g. kvpilot).

    Agents interact with the same interface regardless of mode.
    """

    def __init__(
        self,
        test_run: TestRun,
        runner: BaseRunner,
        gym_server: Optional[Any] = None,
    ):
        self.test_run = test_run
        self.runner = runner
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self._step_count = 0
        self._rng = stdlib_random.Random(42)
        self._trajectory: list[TrajectoryEntry] = []
        self._trajectory_by_iteration: dict[int, list[TrajectoryEntry]] = {}

        if gym_server is not None:
            self._backend: _StepBackend = _GymServerBackend(gym_server)
        elif getattr(test_run.test.cmd_args, "live_rl_mode", False):
            server = _create_gym_server(test_run)
            self._backend = _GymServerBackend(server)
        else:
            self._backend = _RunnerBackend(test_run, runner)

        super().__init__()

    @property
    def _is_online(self) -> bool:
        return isinstance(self._backend, _GymServerBackend)

    def define_action_space(self) -> Dict[str, Any]:
        return self._backend.get_action_space()

    def define_observation_space(self) -> list:
        return self._backend.get_observation_space()

    @property
    def first_sweep(self) -> Any:
        space = self.define_action_space()
        if isinstance(space, dict) and space.get("type") == "continuous":
            shape = int(space.get("shape", 1))
            low = float(space.get("low", -1.0))
            return [low] * shape
        return {k: v[0] if isinstance(v, list) else v for k, v in space.items()}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,  # noqa: Vulture
    ) -> Tuple[list, dict[str, Any]]:
        if seed is not None:
            self._rng = stdlib_random.Random(seed)
        self._step_count = 0
        return self._backend.reset(seed)

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        self._step_count += 1
        observation, done, info = self._backend.step(action)
        reward = self.reward_function(observation)

        entry = TrajectoryEntry(
            step=self._step_count,
            action=action,
            reward=reward,
            observation=observation,
            info=info,
        )
        self._write_trajectory(entry)

        if isinstance(self._backend, _RunnerBackend):
            self._backend.cache_trajectory(entry)

        return observation, reward, done, info

    def render(self, mode: str = "human"):
        if self._is_online:
            logging.info(f"CloudAIGymEnv [online] step {self._step_count}")
        else:
            print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = stdlib_random.Random(seed)
            lazy.np.random.seed(seed)

    def compute_reward(self, observation: list) -> float:
        return self.reward_function(observation)

    def get_observation(self, action: Any) -> list:
        if isinstance(self._backend, _RunnerBackend):
            return self._backend.get_observation(action)
        return self._backend.get_observation_space()

    _MAX_OBS_CSV_ELEMENTS = 1024

    def _write_trajectory(self, entry: TrajectoryEntry) -> None:
        self._trajectory.append(entry)
        self.current_trajectory.append(entry)

        file_exists = self.trajectory_file_path.exists()
        logging.debug(f"Writing trajectory into {self.trajectory_file_path} (exists: {file_exists})")
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trajectory_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation", "info"])
            obs = entry.observation
            if isinstance(obs, list) and len(obs) > self._MAX_OBS_CSV_ELEMENTS:
                obs = f"[truncated len={len(obs)}]"
            writer.writerow([entry.step, entry.action, entry.reward, obs, entry.info])

    def write_trajectory(self, entry: TrajectoryEntry) -> None:
        """Public method for external callers (e.g. single_sbatch_runner)."""
        self._write_trajectory(entry)

    @property
    def trajectory_file_path(self) -> Path:
        if self._is_online:
            return self.test_run.output_path / "trajectory.csv"
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}" / "trajectory.csv"

    @property
    def current_trajectory(self) -> list[TrajectoryEntry]:
        return self._trajectory_by_iteration.setdefault(self.test_run.current_iteration, [])

    def get_cached_trajectory_result(self, action: Any) -> Optional[TrajectoryEntry]:
        for entry in self.current_trajectory:
            if _values_match_exact(entry.action, action):
                return entry
        return None


def _values_match_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    elif isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_match_exact(left[key], right[key]) for key in left)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_match_exact(l, r) for l, r in zip(left, right, strict=True))
    else:
        return left == right
