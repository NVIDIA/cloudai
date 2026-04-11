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

import csv
import dataclasses
import logging
import random as stdlib_random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .base_gym import BaseGym


class GymServer(Protocol):
    """Protocol for gym server objects that CloudAIGymOnlineEnv delegates to."""

    def reset(self) -> Tuple[List[float], Dict[str, Any]]: ...
    def step(self, action: Dict[str, Any]) -> Tuple[List[float], float, bool, Dict[str, Any]]: ...
    def get_action_space(self) -> Dict[str, Any]: ...
    def get_observation_space(self) -> List[float]: ...


@dataclasses.dataclass(frozen=True)
class OnlineTrajectoryEntry:
    """Represents a single step in an online RL trajectory."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list


class CloudAIGymOnlineEnv(BaseGym):
    """
    Gym environment for online RL problems.

    Instead of launching a full workload per step (like CloudAIGymEnv),
    this env delegates to an in-process gym server object for fast,
    stateful interaction.
    """

    def __init__(
        self,
        test_run: Any,
        runner: Any,
        gym_server: Optional[Any] = None,
    ):
        from .._core.registry import Registry

        self.test_run = test_run
        self.runner = runner
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.trajectory: list[OnlineTrajectoryEntry] = []
        self._step_count = 0
        self._rng = stdlib_random.Random(42)

        if gym_server is None:
            server_factory = Registry().get_env_factory("gym_server_" + test_run.test.test_template_name)
            self._server = server_factory(test_run, runner=runner)
        else:
            self._server = gym_server

        super().__init__()

    def define_action_space(self) -> Dict[str, Any]:
        return self._server.get_action_space()

    def define_observation_space(self) -> list:
        return self._server.get_observation_space()

    @property
    def first_sweep(self) -> dict[str, Any]:
        space = self.define_action_space()
        return {k: v[0] if isinstance(v, list) else v for k, v in space.items()}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[list, dict[str, Any]]:
        if seed is not None:
            self._rng = stdlib_random.Random(seed)
        self._step_count = 0
        return self._server.reset()

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        self._step_count += 1
        observation, raw_reward, done, info = self._server.step(action)
        reward = raw_reward if raw_reward != 0.0 else self.reward_function(observation)

        self._write_trajectory(
            OnlineTrajectoryEntry(
                step=self._step_count,
                action=action,
                reward=reward,
                observation=observation,
            )
        )

        return observation, reward, done, info

    def render(self, mode: str = "human"):
        logging.info(f"CloudAIGymOnlineEnv step {self._step_count}")

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = stdlib_random.Random(seed)

    @property
    def trajectory_file_path(self) -> Path:
        return self.runner.scenario_root / self.test_run.name / "online_trajectory.csv"

    def _write_trajectory(self, entry: OnlineTrajectoryEntry):
        self.trajectory.append(entry)
        file_exists = self.trajectory_file_path.exists()
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trajectory_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation"])
            writer.writerow([entry.step, entry.action, entry.reward, entry.observation])
