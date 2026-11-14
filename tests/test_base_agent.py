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

from cloudai.core import BaseAgent, BaseAgentConfig


class _SequenceEnv:
    def __init__(self) -> None:
        self._progress = 0
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
        result = next(self.results)
        if not result[3].get("constraint_rejected", False):
            self._progress += 1
        return result

    @property
    def progress(self) -> int:
        return self._progress


class _RejectingEnv:
    def __init__(self, rejection_limit: int = 3) -> None:
        self._progress = 0
        self._consecutive_rejections = 0
        self.rejection_limit = rejection_limit
        self.skipped_steps = 0

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        return [0.0], {}

    def step(self, _action: dict[str, Any]) -> tuple[list[float], float, bool, dict[str, Any]]:
        self._consecutive_rejections += 1
        skipped = self._consecutive_rejections >= self.rejection_limit
        if skipped:
            self._progress += 1
            self._consecutive_rejections = 0
            self.skipped_steps += 1
        return [-1.0], -1.0, False, {"constraint_rejected": True}

    @property
    def progress(self) -> int:
        return self._progress


class _SeparatedRejectionEnv:
    def __init__(self) -> None:
        self._progress = 0
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
        result = next(self.results)
        if not result[3].get("constraint_rejected", False):
            self._progress += 1
        return result

    @property
    def progress(self) -> int:
        return self._progress


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


def test_run_skips_steps_after_rejected_proposal_limit() -> None:
    env = _RejectingEnv(rejection_limit=3)
    agent = _RecordingAgent(env, BaseAgentConfig(max_consecutive_rejected_proposals=3))

    assert agent.run() == 0

    assert agent.proposals == 6
    assert env.skipped_steps == 2
    assert len(agent.updates) == 6
    assert all(feedback["value"] == -1.0 for feedback in agent.updates)


def test_successful_execution_resets_rejected_proposal_limit() -> None:
    agent = _RecordingAgent(_SeparatedRejectionEnv(), BaseAgentConfig(max_consecutive_rejected_proposals=2))

    assert agent.run() == 0

    assert agent.proposals == 4
    assert [feedback["value"] for feedback in agent.updates] == [-1.0, 1.0, -1.0, 2.0]


def test_run_forwards_the_committed_context_without_expanding_policy_observations() -> None:
    context = {"label": "001", "enabled": True, "optional": None, "empty": {}, "mixed": [1, "2", False]}
    envelope = {
        "env_params": context,
        "configuration_id": "configured-run",
        "run_details": {"run_name": "live", "iteration": 0, "step": 1},
        "measurement": {"kind": "fresh", "outcome": "success"},
        "evaluation_failure": None,
    }
    env = _SequenceEnv()
    env.results = iter(
        [
            ([1.0], 1.0, False, {"env_params": {"sampled": 3}, "trajectory": envelope}),
            ([2.0], 2.0, False, {"env_params": {"sampled": 4}, "trajectory": envelope}),
        ]
    )
    agent = _RecordingAgent(env)

    assert agent.run() == 0

    assert agent.selected_observations == [[0.0], [1.0]]
    assert len(agent.updates) == 2
    assert agent.updates[0]["env_params"] == context
    assert agent.updates[0]["configuration_id"] == "configured-run"
    assert agent.updates[0]["run_details"] == envelope["run_details"]
    assert agent.updates[0]["measurement"] == envelope["measurement"]
    assert agent.updates[0]["evaluation_failure"] is None
    assert agent.updates[0]["action"] == {"candidate": 1}
    assert agent.updates[0]["observation"] == [1.0]
    assert agent.updates[0]["value"] == 1.0
