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
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from cloudai.configurator import BaseGym
from cloudai.core import BaseAgent, BaseAgentConfig, EvaluationResult, Proposal


class _FakeEnv:
    """Like the real env, owns the step allowance and pulls proposals lazily: ``batch_size`` per batch."""

    def __init__(self, batch_size: int = 1) -> None:
        self._progress = 0
        self.batch_size = batch_size
        self.batches = 0
        self.results: list[EvaluationResult] = []

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def evaluate_batch(self, proposals: Iterable[Proposal], max_steps: int | None = None) -> list[EvaluationResult]:
        self.batches += 1
        if max_steps is not None and self._progress >= max_steps:
            return []
        results = [self._evaluate(proposal) for proposal in itertools.islice(proposals, self.batch_size)]
        self.results.extend(results)
        return results

    def _evaluate(self, proposal: Proposal) -> EvaluationResult:
        raise NotImplementedError

    def _result(self, proposal: Proposal, value: float, rejected: bool = False) -> EvaluationResult:
        trial_index, action = proposal
        return EvaluationResult(
            trial_index=trial_index,
            action=action,
            value=value,
            observation=[value],
            done=rejected,
            status="rejected" if rejected else "completed",
            step=None if rejected else self._progress,
            output_path=None,
            env_params={},
        )


class _SequenceEnv(_FakeEnv):
    """Answers proposals with scripted rewards; ``None`` rejects the proposal without consuming a step."""

    def __init__(self, rewards: list[float | None], batch_size: int = 1) -> None:
        super().__init__(batch_size)
        self.rewards = iter(rewards)

    def _evaluate(self, proposal: Proposal) -> EvaluationResult:
        reward = next(self.rewards)
        if reward is None:
            return self._result(proposal, -1.0, rejected=True)
        self._progress += 1
        return self._result(proposal, reward)


class _RejectingEnv(_FakeEnv):
    """Rejects every proposal; each ``rejection_limit``-th rejection in a row consumes a step."""

    def __init__(self, rejection_limit: int = 3) -> None:
        super().__init__()
        self._consecutive_rejections = 0
        self.rejection_limit = rejection_limit
        self.skipped_steps = 0

    def _evaluate(self, proposal: Proposal) -> EvaluationResult:
        self._consecutive_rejections += 1
        if self._consecutive_rejections >= self.rejection_limit:
            self._progress += 1
            self._consecutive_rejections = 0
            self.skipped_steps += 1
        return self._result(proposal, -1.0, rejected=True)


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


class _PairAgent(_RecordingAgent):
    """Proposes two actions before it needs feedback."""

    def select_batch(self, observation: list[float] | None = None) -> Iterator[Proposal]:
        for _ in range(2):
            proposal = self.select_action(observation)
            if proposal is not None:
                yield proposal


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
        {**result, "prev_observation": feedback["prev_observation"]}
        for result, feedback in zip(env.results, agent.updates, strict=True)
    ]


def test_run_skips_steps_after_rejected_proposal_limit() -> None:
    env = _RejectingEnv(rejection_limit=3)
    agent = _RecordingAgent(env, BaseAgentConfig(max_consecutive_rejected_proposals=3))

    assert agent.run() == 0

    assert agent.proposals == 6
    assert env.skipped_steps == 2
    assert len(agent.updates) == 6
    assert all(feedback["value"] == -1.0 for feedback in agent.updates)


def test_successful_execution_resets_rejected_proposal_limit() -> None:
    agent = _RecordingAgent(_SequenceEnv([None, 1.0, None, 2.0]), BaseAgentConfig(max_consecutive_rejected_proposals=2))

    assert agent.run() == 0

    assert agent.proposals == 4
    assert [feedback["value"] for feedback in agent.updates] == [-1.0, 1.0, -1.0, 2.0]


def test_run_shares_prev_observation_within_a_batch() -> None:
    agent = _PairAgent(_SequenceEnv([1.0, 2.0, 3.0, 4.0], batch_size=2))
    agent.max_steps = 4

    assert agent.run() == 0

    assert agent.selected_observations == [[0.0], [0.0], [2.0], [2.0]]
    assert [feedback["trial_index"] for feedback in agent.updates] == [1, 2, 3, 4]
    assert [feedback["prev_observation"] for feedback in agent.updates] == [[0.0], [0.0], [2.0], [2.0]]


def test_default_select_batch_is_lazy() -> None:
    agent = _RecordingAgent(_SequenceEnv([]))

    batch = agent.select_batch([0.0])

    assert agent.proposals == 0
    assert list(batch) == [(1, {"candidate": 1})]
    assert agent.proposals == 1


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


class _StepOnlyEnv(BaseGym):
    """Environment written before batches existed: rejects every second action."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: list[dict[str, Any]] = []

    def define_action_space(self) -> dict[str, Any]:
        return {}

    def define_observation_space(self) -> list:
        return [0.0]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[list, dict[str, Any]]:
        del seed, options
        return [0.0], {}

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        self.actions.append(action)
        rejected = len(self.actions) % 2 == 0
        value = -1.0 if rejected else float(len(self.actions))
        return [value], value, rejected, {}

    def render(self, mode: str = "human") -> None:
        del mode

    def seed(self, seed: int | None = None) -> None:
        del seed


def test_run_drives_an_environment_that_only_implements_step() -> None:
    env = _StepOnlyEnv()
    agent = _RecordingAgent(env)  # type: ignore[arg-type]

    assert agent.run() == 0

    assert agent.proposals == 3
    assert [update["status"] for update in agent.updates] == ["completed", "rejected", "completed"]
    assert [update["step"] for update in agent.updates] == [1, None, 2]
    assert [update["value"] for update in agent.updates] == [1.0, -1.0, 3.0]
    assert agent.selected_observations == [[0.0], [1.0], [-1.0]]


def test_max_parallelism_must_be_positive() -> None:
    assert BaseAgentConfig().max_parallelism == 1
    with pytest.raises(ValidationError):
        BaseAgentConfig(max_parallelism=0)
