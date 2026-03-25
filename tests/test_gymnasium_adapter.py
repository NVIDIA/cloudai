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

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import patch

import gymnasium
import numpy as np
import pytest

from cloudai.configurator.base_gym import BaseGym
from cloudai.configurator.gymnasium_adapter import GymnasiumAdapter


class _FakeGym(BaseGym):
    """Minimal BaseGym implementation with a known, deterministic interface."""

    def __init__(self) -> None:
        self._action_space: dict[str, Any] = {
            "param_a": [1, 2, 3],
            "param_b": [10, 20],
        }
        self._observation_space: list[float] = [0.0, 0.0, 0.0]
        super().__init__()

    def define_action_space(self) -> dict[str, Any]:
        return self._action_space

    def define_observation_space(self) -> list:
        return self._observation_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[list, dict[str, Any]]:
        return [0.0, 0.0, 0.0], {}

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        return [1.0, 2.0, 3.0], 0.5, False, {"info": "test"}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


@pytest.fixture
def fake_gym() -> _FakeGym:
    return _FakeGym()


@pytest.fixture
def adapter(fake_gym: _FakeGym) -> GymnasiumAdapter:
    return GymnasiumAdapter(fake_gym)


def test_adapter_action_space_structure(adapter: GymnasiumAdapter) -> None:
    assert isinstance(adapter.action_space, gymnasium.spaces.Dict)

    assert "param_a" in adapter.action_space.spaces
    assert "param_b" in adapter.action_space.spaces

    space_a = adapter.action_space.spaces["param_a"]
    space_b = adapter.action_space.spaces["param_b"]

    assert isinstance(space_a, gymnasium.spaces.Discrete)
    assert isinstance(space_b, gymnasium.spaces.Discrete)

    assert space_a.n == 3
    assert space_b.n == 2


def test_adapter_observation_space_structure(adapter: GymnasiumAdapter) -> None:
    assert isinstance(adapter.observation_space, gymnasium.spaces.Box)
    assert adapter.observation_space.shape == (3,)
    assert adapter.observation_space.dtype == np.float32


def test_adapter_reset_returns_numpy_array(adapter: GymnasiumAdapter) -> None:
    obs, info = adapter.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (3,)
    np.testing.assert_array_equal(obs, [0.0, 0.0, 0.0])
    assert isinstance(info, dict)


def test_adapter_step_returns_five_tuple(adapter: GymnasiumAdapter) -> None:
    adapter.reset()
    result = adapter.step({"param_a": 0, "param_b": 1})

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(obs, [1.0, 2.0, 3.0])

    assert isinstance(reward, float)
    assert reward == 0.5

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert isinstance(info, dict)


def test_adapter_decode_action(adapter: GymnasiumAdapter) -> None:
    decoded = adapter.decode_action({"param_a": 0, "param_b": 1})
    assert decoded == {"param_a": 1, "param_b": 20}


def test_adapter_single_value_params_excluded() -> None:
    """Params with a single value offer no choice and should be excluded."""

    class _SingleValueGym(_FakeGym):
        def define_action_space(self) -> dict[str, Any]:
            return {
                "param_a": [1, 2, 3],
                "param_b": [10, 20],
                "fixed_param": [42],
            }

    adapter = GymnasiumAdapter(_SingleValueGym())

    assert "fixed_param" not in adapter.action_space.spaces
    assert "param_a" in adapter.action_space.spaces
    assert "param_b" in adapter.action_space.spaces


def test_adapter_import_error_without_gymnasium() -> None:
    with (
        patch(
            "cloudai.configurator.gymnasium_adapter._import_gymnasium",
            side_effect=ImportError(
                "gymnasium is required for GymnasiumAdapter. Install it with: pip install gymnasium"
            ),
        ),
        pytest.raises(ImportError, match="pip install gymnasium"),
    ):
        GymnasiumAdapter(_FakeGym())


def test_adapter_unwrapped_returns_original(fake_gym: _FakeGym, adapter: GymnasiumAdapter) -> None:
    assert adapter.unwrapped is fake_gym
