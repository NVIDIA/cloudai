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

from types import SimpleNamespace
from typing import Any, Optional

import gymnasium
import numpy as np
import pytest

from cloudai.configurator import GymnasiumAdapter
from cloudai.configurator.base_gym import BaseGym


class _FakeGym(BaseGym):
    """Deterministic BaseGym fixture with two tunable params and a 3-dim observation."""

    def __init__(self) -> None:
        self._action_space: dict[str, Any] = {"param_a": [1, 2, 3], "param_b": [10, 20]}
        self._observation_space: list[float] = [0.0, 0.0, 0.0]
        self.last_action: Optional[dict[str, Any]] = None
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
        self.last_action = action
        return [1.0, 2.0, 3.0], 0.5, False, {"info": "test"}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class _FixedParamGym(_FakeGym):
    """Adds a single-value parameter that the adapter must treat as fixed."""

    def define_action_space(self) -> dict[str, Any]:
        return {"param_a": [1, 2, 3], "param_b": [10, 20], "fixed_param": [42]}


class _GymWithTestRun(_FakeGym):
    """Carries a CloudAI-like ``test_run`` so we can verify the step-counter sync."""

    def __init__(self) -> None:
        super().__init__()
        self.test_run = SimpleNamespace(step=0)


@pytest.fixture
def fake_gym() -> _FakeGym:
    return _FakeGym()


@pytest.fixture
def adapter(fake_gym: _FakeGym) -> GymnasiumAdapter:
    return GymnasiumAdapter(fake_gym)


def test_action_space_is_dict_of_discrete(adapter: GymnasiumAdapter) -> None:
    assert isinstance(adapter.action_space, gymnasium.spaces.Dict)
    assert set(adapter.action_space.spaces) == {"param_a", "param_b"}

    sub_a = adapter.action_space.spaces["param_a"]
    sub_b = adapter.action_space.spaces["param_b"]
    assert isinstance(sub_a, gymnasium.spaces.Discrete) and sub_a.n == 3
    assert isinstance(sub_b, gymnasium.spaces.Discrete) and sub_b.n == 2


def test_observation_space_shape_matches_env(adapter: GymnasiumAdapter) -> None:
    assert isinstance(adapter.observation_space, gymnasium.spaces.Box)
    assert adapter.observation_space.shape == (3,)
    assert adapter.observation_space.dtype == np.float32


def test_reset_returns_float32_array(adapter: GymnasiumAdapter) -> None:
    obs, info = adapter.reset()
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32 and obs.shape == (3,)
    np.testing.assert_array_equal(obs, [0.0, 0.0, 0.0])
    assert info == {}


def test_step_returns_gymnasium_five_tuple(adapter: GymnasiumAdapter) -> None:
    adapter.reset()
    obs, reward, terminated, truncated, info = adapter.step({"param_a": 0, "param_b": 1})

    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs, [1.0, 2.0, 3.0])
    assert reward == 0.5
    assert terminated is False
    assert truncated is False
    assert info == {"info": "test"}


def test_decode_action_maps_indices_back_to_values(adapter: GymnasiumAdapter) -> None:
    assert adapter.decode_action({"param_a": 0, "param_b": 1}) == {"param_a": 1, "param_b": 20}


def test_unwrapped_returns_original_env(fake_gym: _FakeGym, adapter: GymnasiumAdapter) -> None:
    assert adapter.unwrapped is fake_gym


def test_single_value_params_are_excluded_from_action_space() -> None:
    adapter = GymnasiumAdapter(_FixedParamGym())

    assert set(adapter.action_space.spaces) == {"param_a", "param_b"}
    assert adapter._fixed_params == {"fixed_param": 42}


def test_step_merges_fixed_params_into_underlying_action() -> None:
    gym = _FixedParamGym()
    adapter = GymnasiumAdapter(gym)
    adapter.reset()

    adapter.step({"param_a": 0, "param_b": 1})

    assert gym.last_action == {"param_a": 1, "param_b": 20, "fixed_param": 42}


def test_step_raw_bypasses_decode_and_fixed_injection() -> None:
    gym = _FixedParamGym()
    adapter = GymnasiumAdapter(gym)
    adapter.reset()
    raw = {"param_a": 999, "param_b": 888, "fixed_param": 777}

    obs, _reward, terminated, truncated, _info = adapter.step_raw(raw)

    assert gym.last_action == raw
    assert isinstance(obs, np.ndarray)
    assert terminated is False
    assert truncated is False


def test_step_assigns_one_based_step_to_test_run() -> None:
    gym = _GymWithTestRun()
    adapter = GymnasiumAdapter(gym)
    adapter.reset()

    adapter.step({"param_a": 0, "param_b": 1})
    assert gym.test_run.step == 1

    adapter.step({"param_a": 1, "param_b": 0})
    assert gym.test_run.step == 2


def test_step_raw_also_syncs_test_run_step() -> None:
    gym = _GymWithTestRun()
    adapter = GymnasiumAdapter(gym)
    adapter.reset()

    adapter.step_raw({"param_a": 2, "param_b": 1})
    assert gym.test_run.step == 1


def test_reset_restarts_step_counter() -> None:
    gym = _GymWithTestRun()
    adapter = GymnasiumAdapter(gym)
    adapter.reset()
    adapter.step({"param_a": 0, "param_b": 1})
    adapter.step({"param_a": 1, "param_b": 0})
    assert gym.test_run.step == 2

    adapter.reset()
    adapter.step({"param_a": 0, "param_b": 0})
    assert gym.test_run.step == 1


def test_missing_gymnasium_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import cloudai.configurator.gymnasium_adapter as mod

    def _raise() -> None:
        raise ImportError("pip install gymnasium")

    monkeypatch.setattr(mod, "_import_gymnasium", _raise)

    with pytest.raises(ImportError, match="pip install gymnasium"):
        GymnasiumAdapter(_FakeGym())
