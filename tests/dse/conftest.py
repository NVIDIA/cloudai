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

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from cloudai.core import BaseAgent, BaseAgentConfig, Proposal, Registry
from cloudai.systems.standalone import StandaloneSystem

from .probe import ProbeStandaloneCommandGenStrategy, ProbeTestDefinition, ScriptedSystem, scripted_system


class ProbeAgent(BaseAgent):
    """Base of the test agents: cycles through ``x`` and records everything it is told."""

    created: ClassVar[list[ProbeAgent]] = []

    def __init__(self, env: Any, config: BaseAgentConfig) -> None:
        super().__init__(env, config)
        self.max_steps = env.max_steps
        self.xs: list[int] = env.define_action_space()["x"]
        self.proposed = 0
        self.log: list[tuple[str, int]] = []
        self.updates: list[dict[str, Any]] = []
        ProbeAgent.created.append(self)

    @staticmethod
    def get_config_class() -> type[BaseAgentConfig]:
        return BaseAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        del config

    def propose(self) -> Proposal:
        self.proposed += 1
        self.log.append(("propose", self.proposed))
        return self.proposed, {"x": self.xs[(self.proposed - 1) % len(self.xs)]}

    def select_action(self, observation: list[float] | None = None) -> Proposal | None:
        del observation
        return self.propose()

    def update_policy(self, _feedback: dict[str, Any]) -> None:
        self.log.append(("update", _feedback["trial_index"]))
        self.updates.append(_feedback)


class SerialAgent(ProbeAgent):
    """Proposes one action at a time, like every agent written before batches existed."""


class StreamAgent(ProbeAgent):
    """Needs no feedback: an endless stream of proposals, bounded only by the step allowance."""

    supports_variable_environment = True

    def select_batch(self, observation: list[float] | None = None) -> Iterator[Proposal]:
        del observation
        while True:
            yield self.propose()


class GenerationAgent(ProbeAgent):
    """Proposes three actions at a time and needs their feedback before the next three."""

    def select_batch(self, observation: list[float] | None = None) -> Iterator[Proposal]:
        del observation
        for _ in range(3):
            yield self.propose()


@pytest.fixture(autouse=True)
def probe_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry()
    monkeypatch.setitem(registry.agents_map, "probe_serial", SerialAgent)
    monkeypatch.setitem(registry.agents_map, "probe_stream", StreamAgent)
    monkeypatch.setitem(registry.agents_map, "probe_generation", GenerationAgent)
    monkeypatch.setitem(
        registry.command_gen_strategies_map,
        (StandaloneSystem, ProbeTestDefinition),
        ProbeStandaloneCommandGenStrategy,
    )
    monkeypatch.setattr(ProbeAgent, "created", [])


@pytest.fixture
def system(tmp_path: Path) -> ScriptedSystem:
    return scripted_system(tmp_path)
