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

from typing import Any

import pytest

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


class _RejectingEnv:
    def __init__(self) -> None:
        self.skipped_steps = 0

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def step(self, _action: dict[str, Any]) -> tuple[list[float], float, bool, dict[str, Any]]:
        return [-1.0], -1.0, False, {"constraint_rejected": True}

    def skip_step(self) -> None:
        self.skipped_steps += 1


class _SeparatedRejectionEnv:
    def __init__(self) -> None:
        self.results = iter(
            [
                ([-1.0], -1.0, False, {"constraint_rejected": True}),
                ([1.0], 1.0, False, {}),
                ([-1.0], -1.0, False, {"constraint_rejected": True}),
                ([2.0], 2.0, False, {}),
            ]
        )

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def step(self, _action: dict[str, Any]) -> tuple[list[float], float, bool, dict[str, Any]]:
        return next(self.results)


class _RecordingAgent(BaseAgent):
    def __init__(
        self,
        env: _SequenceEnv | _RejectingEnv | _SeparatedRejectionEnv,
        config: BaseAgentConfig | None = None,
    ) -> None:
        super().__init__(env, config or BaseAgentConfig())  # type: ignore[arg-type]
        self.max_steps = 2
        self.proposals = 0
        self.selected_observations: list[list[float] | None] = []
        self.updates: list[dict[str, Any]] = []

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


def test_run_retries_constraint_rejection_without_consuming_execution_step() -> None:
    agent = _RecordingAgent(_SequenceEnv())

    assert agent.run() == 0

    assert agent.proposals == 3
    assert agent.selected_observations == [[0.0], [-1.0], [2.0]]
    assert [feedback["trial_index"] for feedback in agent.updates] == [1, 2, 3]
    assert [feedback["value"] for feedback in agent.updates] == [-1.0, 2.0, 3.0]
    assert "constraint_rejected" not in agent.updates[0]
    assert "dse_step_skipped" not in agent.updates[0]
    assert all("execution_step" not in feedback for feedback in agent.updates)


def test_run_skips_steps_after_rejected_proposal_limit(caplog: pytest.LogCaptureFixture) -> None:
    env = _RejectingEnv()
    agent = _RecordingAgent(env, BaseAgentConfig(max_consecutive_rejected_proposals=3))

    with caplog.at_level("INFO"):
        assert agent.run() == 0

    assert agent.proposals == 6
    assert env.skipped_steps == 2
    assert len(agent.updates) == 6
    assert all(feedback["value"] == -1.0 for feedback in agent.updates)
    assert "Skipping DSE step 1/2 after repeated constraint rejections" in caplog.text
    assert "Skipping DSE step 2/2 after repeated constraint rejections" in caplog.text


def test_successful_execution_resets_rejected_proposal_limit() -> None:
    agent = _RecordingAgent(_SeparatedRejectionEnv(), BaseAgentConfig(max_consecutive_rejected_proposals=2))

    assert agent.run() == 0

    assert agent.proposals == 4
    assert [feedback["value"] for feedback in agent.updates] == [-1.0, 1.0, -1.0, 2.0]
