# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
from collections.abc import Iterable
from typing import Any

import pytest
from pydantic import ValidationError

from cloudai.core import BaseAgent, BaseAgentConfig, Proposal, Trial


class _FakeEnv:
    """Like the real env, owns the step allowance and pulls one proposal per batch."""

    def __init__(self) -> None:
        self._progress = 0
        self.batches = 0
        self.results: list[Trial] = []

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def evaluate_batch(self, proposals: Iterable[Proposal], max_steps: int | None = None) -> list[Trial]:
        self.batches += 1
        if max_steps is not None and self._progress >= max_steps:
            return []
        results = [self._evaluate(proposal) for proposal in itertools.islice(proposals, 1)]
        self.results.extend(results)
        return results

    def _evaluate(self, proposal: Proposal) -> Trial:
        raise NotImplementedError

    def _result(self, proposal: Proposal, value: float, rejected: bool = False) -> Trial:
        trial_index, action = proposal
        return Trial(
            trial_index=trial_index,
            action=action,
            reward=value,
            observation=[value],
            status="rejected" if rejected else "completed",
            step=None if rejected else self._progress,
            output_path=None,
            env_params={},
        )


class _SequenceEnv(_FakeEnv):
    """Answers proposals with scripted rewards; ``None`` rejects the proposal without consuming a step."""

    def __init__(self, rewards: list[float | None]) -> None:
        super().__init__()
        self.rewards = iter(rewards)

    def _evaluate(self, proposal: Proposal) -> Trial:
        reward = next(self.rewards)
        if reward is None:
            return self._result(proposal, -1.0, rejected=True)
        self._progress += 1
        return self._result(proposal, reward)


class _RecordingAgent(BaseAgent):
    def __init__(
        self,
        env: _FakeEnv,
        config: BaseAgentConfig | None = None,
        max_proposals: int | None = None,
    ) -> None:
        super().__init__(env, config or BaseAgentConfig())  # type: ignore[arg-type]
        self.max_steps = 2
        self.max_proposals = max_proposals
        self.proposals = 0
        self.selected_observations: list[list[float] | None] = []
        self.updates: list[dict[str, Any]] = []

    @staticmethod
    def get_config_class() -> type[BaseAgentConfig]:
        return BaseAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        del config

    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]] | None:
        self.selected_observations.append(observation)
        if self.proposals == self.max_proposals:
            return None
        self.proposals += 1
        return self.proposals, {"candidate": self.proposals}

    def update_policy(self, feedback: dict[str, Any]) -> None:
        self.updates.append(feedback)


def test_run_retries_constraint_rejection_without_consuming_execution_step() -> None:
    env = _SequenceEnv([None, 2.0, 3.0])
    agent = _RecordingAgent(env)

    assert agent.run() == 0

    assert agent.proposals == 3
    assert agent.selected_observations == [[0.0], [-1.0], [2.0]]
    assert [feedback["trial_index"] for feedback in agent.updates] == [1, 2, 3]
    assert [feedback["value"] for feedback in agent.updates] == [-1.0, 2.0, 3.0]
    assert [feedback["done"] for feedback in agent.updates] == [True, False, False]
    assert [feedback["action"] for feedback in agent.updates] == [{"candidate": 1}, {"candidate": 2}, {"candidate": 3}]
    assert [feedback["observation"] for feedback in agent.updates] == [[-1.0], [2.0], [3.0]]
    assert [feedback["prev_observation"] for feedback in agent.updates] == [[0.0], [-1.0], [2.0]]
    assert agent.updates == [
        {**result.as_feedback(), "prev_observation": feedback["prev_observation"]}
        for result, feedback in zip(env.results, agent.updates, strict=True)
    ]


def test_run_does_not_select_action_when_env_does_not_pull() -> None:
    env = _SequenceEnv([1.0])
    agent = _RecordingAgent(env)
    agent.max_steps = 0

    assert agent.run() == 0

    assert env.batches == 1
    assert agent.selected_observations == []
    assert agent.updates == []


def test_run_stops_when_select_action_returns_none() -> None:
    env = _SequenceEnv([1.0, 2.0, 3.0])
    agent = _RecordingAgent(env, max_proposals=2)
    agent.max_steps = 10

    assert agent.run() == 0

    assert agent.selected_observations == [[0.0], [1.0], [2.0]]
    assert env.batches == 3
    assert [feedback["value"] for feedback in agent.updates] == [1.0, 2.0]


def test_max_parallelism_must_be_positive() -> None:
    assert BaseAgentConfig().max_parallelism == 1
    with pytest.raises(ValidationError):
        BaseAgentConfig(max_parallelism=0)
