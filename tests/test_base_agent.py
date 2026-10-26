# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from cloudai.core import BaseAgent, BaseAgentConfig


class _SequenceEnv:
    def __init__(self) -> None:
        self.results = iter(
            [
                ([-1.0], -1.0, False, {"constraint_rejected": True}),
                ([2.0], 2.0, False, {}),
                ([3.0], 3.0, False, {}),
            ]
        )

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def step(self, _action: dict[str, Any]) -> tuple[list[float], float, bool, dict[str, Any]]:
        return next(self.results)


class _RecordingAgent(BaseAgent):
    def __init__(self, env: _SequenceEnv) -> None:
        super().__init__(env, BaseAgentConfig())  # type: ignore[arg-type]
        self.max_steps = 2
        self.proposals = 0
        self.selected_observations: list[list[float] | None] = []
        self.updates: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []

    @staticmethod
    def get_config_class() -> type[BaseAgentConfig]:
        return BaseAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        del config

    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]]:
        self.selected_observations.append(observation)
        self.proposals += 1
        return self.proposals, {"candidate": self.proposals}

    def update_policy(self, feedback: dict[str, Any]) -> None:
        self.updates.append(feedback)

    def discard_action(self, feedback: dict[str, Any]) -> None:
        self.discards.append(feedback)


def test_run_retries_constraint_rejection_and_only_learns_from_success() -> None:
    agent = _RecordingAgent(_SequenceEnv())

    assert agent.run() == 0

    assert agent.proposals == 3
    assert agent.selected_observations == [[0.0], [0.0], [2.0]]
    assert [feedback["trial_index"] for feedback in agent.updates] == [2, 3]
    assert agent.updates[0]["value"] == 2.0
    assert len(agent.discards) == 1
    assert all(feedback["value"] is None for feedback in agent.discards)
    assert [feedback["execution_step"] for feedback in agent.discards] == [0]
