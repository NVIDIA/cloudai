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

from pathlib import Path

import pytest

from cloudai.core import Registry
from cloudai.systems.standalone import StandaloneSystem
from tests.configurator.probe import (
    GenerationAgent,
    ProbeAgent,
    ProbeStandaloneCommandGenStrategy,
    ProbeTestDefinition,
    ScriptedSystem,
    SerialAgent,
    StreamAgent,
    scripted_system,
)


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
