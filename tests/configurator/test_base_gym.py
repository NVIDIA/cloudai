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

from cloudai.configurator import BaseGym
from cloudai.core import Trial


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


def test_step_only_environment_batch_fallback():
    env = _StepOnlyEnv()
    trials = env.evaluate_batch(((i, {"x": i}) for i in range(1, 5)), max_steps=2)
    assert [(t.trial_index, t.status, t.step, t.reward) for t in trials] == [
        (1, "completed", 1, 1.0),
        (2, "rejected", None, -1.0),
        (3, "completed", 2, 3.0),
    ]
    assert env.evaluate_batch([(4, {"x": 4})], max_steps=2) == []


def test_pending_trial_cannot_be_sent_to_an_agent():
    with pytest.raises(ValueError, match="pending trial"):
        Trial(trial_index=1, action={"x": 1}).as_feedback()
