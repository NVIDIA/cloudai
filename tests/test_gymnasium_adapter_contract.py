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

"""
Caller-contract tests for ``GymnasiumAdapter``.

The single invariant every consumer assumes:

    ``test_run.step`` is a **monotonic trial index** across the entire run.

Gym's ``reset()`` is an *episode boundary*, not a trial boundary. For the
contextual-bandit configs (``agent_steps=1``), RLlib calls ``reset()`` before
*every* trial. An earlier adapter rewound ``test_run.step`` on reset and
collapsed every trial onto step 1 — silently overwriting output dirs and
producing duplicate-step rows in trajectory.csv / env.csv.

These tests pin the negative invariant: the adapter must not mutate
``test_run.step``. That counter is owned by ``TestRun`` and advanced
exclusively by ``CloudAIGymEnv.step()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from cloudai._core.action_space import ContinuousSpace
from cloudai.configurator.base_gym import BaseGym
from cloudai.configurator.env_params import ObsLeafDescriptor

try:
    import gymnasium  # noqa: F401

    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False

pytestmark = pytest.mark.skipif(not _HAS_GYMNASIUM, reason="gymnasium not installed")

from cloudai.configurator.gymnasium_adapter import GymnasiumAdapter  # noqa: E402


class _StubBaseGym(BaseGym):
    """Minimal BaseGym with a ``test_run`` attribute mirroring CloudAIGymEnv."""

    def __init__(self) -> None:
        self._action_space: dict[str, Any] = {"param_a": [1, 2, 3], "param_b": [10, 20]}
        self._observation_space: list[float] = [0.0, 0.0, 0.0]
        self.test_run = SimpleNamespace(step=0)
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
        self.test_run.step += 1
        return [1.0, 2.0, 3.0], 0.5, False, {}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class TestStepIsMonotonicTrialIndex:
    """``test_run.step`` is a trial index, not an episode-local counter."""

    def test_step_advances_within_single_episode(self) -> None:
        gym = _StubBaseGym()
        adapter = GymnasiumAdapter(gym)
        adapter.reset()

        seen: list[int] = []
        for _ in range(3):
            adapter.step({"param_a": 0, "param_b": 0})
            seen.append(gym.test_run.step)

        assert seen == [1, 2, 3]

    def test_step_is_monotonic_across_episode_boundaries(self) -> None:
        """The bug: ``reset()`` rewinds ``_step_count`` to 0, so the next
        ``step()`` writes ``test_run.step = 1`` again. With contextual-bandit
        RLlib (one step per episode) this means every trial reports step 1.
        """
        gym = _StubBaseGym()
        adapter = GymnasiumAdapter(gym)

        seen: list[int] = []
        for _ in range(5):
            adapter.reset()
            adapter.step({"param_a": 0, "param_b": 0})
            seen.append(gym.test_run.step)

        assert seen == [1, 2, 3, 4, 5], (
            f"test_run.step must be a monotonic trial index across episodes; got {seen}. "
            "reset() is a Gym episode boundary, not a trial boundary; rewinding the "
            "trial counter collapses every contextual-bandit rollout onto step 1."
        )

    def test_mixed_within_and_across_episode_steps_are_monotonic(self) -> None:
        gym = _StubBaseGym()
        adapter = GymnasiumAdapter(gym)

        seen: list[int] = []
        for episode_len in (2, 1, 3):
            adapter.reset()
            for _ in range(episode_len):
                adapter.step({"param_a": 0, "param_b": 0})
                seen.append(gym.test_run.step)

        assert seen == [1, 2, 3, 4, 5, 6], (
            f"test_run.step must be a monotonic trial index regardless of episode shape; got {seen}"
        )


class _ContextualStubBaseGym(BaseGym):
    """BaseGym stub that simulates CloudAIGymEnv's contextual-obs contract.

    ``reset()`` returns an observation with a per-trial context value in
    slot 1 (mimicking how the upstream env writes a sampled env_param into
    the obs vector built at the trial boundary). Each call to ``reset()``
    advances the simulated trial counter so we can assert the adapter
    surfaces the *current* context, not a stale one.
    """

    def __init__(self, contexts: list[float]) -> None:
        self._contexts = list(contexts)
        self._action_space: dict[str, Any] = {"param_a": [1, 2, 3], "param_b": [10, 20]}
        self.test_run = SimpleNamespace(step=0)
        super().__init__()

    def define_action_space(self) -> dict[str, Any]:
        return self._action_space

    def define_observation_space(self) -> list:
        return [0.0, 0.0]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[list, dict[str, Any]]:
        ctx = self._contexts[self.test_run.step]
        self.test_run.step += 1
        return [0.0, ctx], {}

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        ctx = self._contexts[self.test_run.step - 1]
        return [42.0, ctx], 0.5, False, {}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class TestAdapterPropagatesContextualObservation:
    """The adapter must pass through env-built observations unchanged.

    With the contextual-bandit fix in cloudai, ``CloudAIGymEnv.reset()``
    samples env_params at the trial boundary and bakes them into the obs
    vector before returning. RLlib's policy reads obs from
    ``adapter.reset()``, so the adapter must propagate that vector verbatim
    (modulo numpy-float32 casting). The same propagation invariant applies
    on ``adapter.step()``.
    """

    def test_reset_propagates_context_into_observation(self) -> None:
        contexts = [0.001, 0.0, 0.01, 0.001]
        gym = _ContextualStubBaseGym(contexts)
        adapter = GymnasiumAdapter(gym)

        seen: list[float] = []
        for _ in range(len(contexts)):
            obs, _info = adapter.reset()
            seen.append(float(obs[1]))

        assert seen == pytest.approx(contexts, rel=1e-5), (
            f"adapter.reset() must surface the trial's context value (slot 1) verbatim "
            f"(modulo float32 cast); got {seen}, expected {contexts}"
        )

    def test_step_propagates_context_into_observation(self) -> None:
        contexts = [0.0, 0.01, 0.001]
        gym = _ContextualStubBaseGym(contexts)
        adapter = GymnasiumAdapter(gym)

        for ctx in contexts:
            adapter.reset()
            obs, _r, _term, _trunc, _info = adapter.step({"param_a": 0, "param_b": 0})
            assert float(obs[0]) == pytest.approx(42.0, rel=1e-5), (
                f"adapter.step() must propagate the env's measured-metric slot; got {obs[0]}"
            )
            assert float(obs[1]) == pytest.approx(ctx, rel=1e-5), (
                f"adapter.step() must propagate the trial's context value; got {obs[1]}, expected {ctx}"
            )


class _ContinuousStubBaseGym(BaseGym):
    """BaseGym stub that surfaces a ContinuousSpace in its action_space.

    Records the params received by ``step`` so the test can assert what the
    adapter actually emits to the env (post-rounding / clamping).
    """

    def __init__(self, action_space: dict[str, Any]) -> None:
        self._action_space = action_space
        self.test_run = SimpleNamespace(step=0)
        self.received: list[dict[str, Any]] = []
        super().__init__()

    def define_action_space(self) -> dict[str, Any]:
        return self._action_space

    def define_observation_space(self) -> list:
        return [0.0]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[list, dict[str, Any]]:
        return [0.0], {}

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        self.received.append(dict(action))
        return [1.0], 0.5, False, {}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class TestAdapterDispatchesContinuousSpace:
    """``ContinuousSpace`` in cloudai's param_space → ``gym.spaces.Box`` in the adapter.

    The adapter is also responsible for **actuator quantization**: when the
    spec is ``dtype="int"``, the float coming out of the policy is rounded at
    decode_action, so the env (and trajectory cache) sees only ints. This
    collapses float jitter (47.34 vs 47.36) onto the same emitted int and
    keeps the cache hit-rate honest.
    """

    @staticmethod
    def _action_space() -> dict[str, Any]:
        return {
            "threshold": ContinuousSpace(low=0.0, high=200.0, dtype="int"),
            "discrete": [10, 20, 30],
            "fixed": [99],
        }

    def test_action_space_uses_box_for_continuous_and_discrete_for_list(self) -> None:
        import gymnasium

        gym_env = _ContinuousStubBaseGym(self._action_space())
        adapter = GymnasiumAdapter(gym_env)

        threshold = adapter.action_space["threshold"]
        assert isinstance(threshold, gymnasium.spaces.Box)
        assert threshold.shape == (1,)
        assert float(threshold.low[0]) == pytest.approx(0.0)
        assert float(threshold.high[0]) == pytest.approx(200.0)

        discrete = adapter.action_space["discrete"]
        assert isinstance(discrete, gymnasium.spaces.Discrete)
        assert int(discrete.n) == 3

        assert "fixed" not in adapter.action_space, "single-element lists are fixed, not tunable"

    def test_decode_action_rounds_dtype_int(self) -> None:
        import numpy as np

        gym_env = _ContinuousStubBaseGym(self._action_space())
        adapter = GymnasiumAdapter(gym_env)

        decoded = adapter.decode_action({"threshold": np.array([47.34], dtype=np.float32), "discrete": 1})

        assert decoded["threshold"] == 47, f"dtype=int must round; got {decoded['threshold']}"
        assert isinstance(decoded["threshold"], int), "rounded action must be Python int, not float"
        assert decoded["discrete"] == 20, "Discrete index 1 → 20"

    @pytest.mark.parametrize("raw,expected", [(47.34, 47), (47.36, 47), (47.5, 48), (47.4999, 47)])
    def test_cache_key_collapses_float_jitter_to_same_int(self, raw: float, expected: int) -> None:
        """Adjacent float actions that round to the same int must collapse identically.

        This is the actuator-quantization invariant from the design doc: the
        env (and trajectory cache key) must see the same int for any float in
        the rounding interval, so the cache fills with semantic duplicates,
        not float-noise duplicates.
        """
        import numpy as np

        gym_env = _ContinuousStubBaseGym(self._action_space())
        adapter = GymnasiumAdapter(gym_env)

        decoded = adapter.decode_action({"threshold": np.array([raw], dtype=np.float32), "discrete": 0})
        assert decoded["threshold"] == expected

    @pytest.mark.parametrize("raw,expected", [(-5.0, 0), (250.0, 200), (0.0, 0), (200.0, 200)])
    def test_decode_action_clamps_to_range(self, raw: float, expected: int) -> None:
        """Out-of-range continuous actions clamp to ``[low, high]``."""
        import numpy as np

        gym_env = _ContinuousStubBaseGym(self._action_space())
        adapter = GymnasiumAdapter(gym_env)

        decoded = adapter.decode_action({"threshold": np.array([raw], dtype=np.float32), "discrete": 0})
        assert decoded["threshold"] == expected

    def test_step_emits_rounded_int_to_underlying_env(self) -> None:
        """The env (and downstream cache) must see the *rounded* int, not the raw float."""
        import numpy as np

        gym_env = _ContinuousStubBaseGym(self._action_space())
        adapter = GymnasiumAdapter(gym_env)
        adapter.reset()

        adapter.step({"threshold": np.array([78.6], dtype=np.float32), "discrete": 2})

        assert gym_env.received, "env.step was not called"
        emitted = gym_env.received[-1]
        assert emitted["threshold"] == 79, "env must receive rounded int, not raw float"
        assert emitted["discrete"] == 30, "Discrete decode unaffected by continuous-action plumbing"
        assert emitted["fixed"] == 99, "fixed params must be injected by the adapter"

    def test_dtype_float_preserves_continuous_value(self) -> None:
        """``dtype="float"`` clamps but does NOT round; the policy's float reaches the env."""
        import numpy as np

        action_space: dict[str, Any] = {
            "knob": ContinuousSpace(low=0.0, high=1.0, dtype="float"),
        }
        gym_env = _ContinuousStubBaseGym(action_space)
        adapter = GymnasiumAdapter(gym_env)

        decoded = adapter.decode_action({"knob": np.array([0.7234], dtype=np.float32)})
        assert decoded["knob"] == pytest.approx(0.7234, rel=1e-4)
        assert isinstance(decoded["knob"], float)


class _StructuredStubBaseGym(BaseGym):
    """BaseGym stub that opts in/out of the structured (Dict) obs path.

    Mirrors ``CloudAIGymEnv``'s gate: ``structured_observation_descriptors``
    returns ``None`` for a metrics-only env (no observed name is a declared
    env_param) and a per-leaf descriptor dict otherwise. ``encode_observation``
    produces the matching encoded leaves.
    """

    def __init__(self, descriptors: Optional[dict[str, ObsLeafDescriptor]], obs_dim: int = 2) -> None:
        self._descriptors = descriptors
        self._obs_dim = obs_dim
        self._action_space: dict[str, Any] = {"param_a": [1, 2, 3]}
        self.test_run = SimpleNamespace(step=0)
        super().__init__()

    def define_action_space(self) -> dict[str, Any]:
        return self._action_space

    def define_observation_space(self) -> list:
        return [0.0] * self._obs_dim

    def structured_observation_descriptors(self) -> Optional[dict[str, ObsLeafDescriptor]]:
        return self._descriptors

    def encode_observation(self, raw_values: list) -> dict[str, Any]:
        out: dict[str, Any] = {}
        assert self._descriptors is not None
        for i, (name, desc) in enumerate(self._descriptors.items()):
            raw = raw_values[i] if i < len(raw_values) else 0.0
            if desc.kind == "discrete":
                out[name] = int(raw)
            elif desc.dim == 2:
                out[name] = [1.0 if raw <= 0.0 else 0.0, float(raw)]
            else:
                out[name] = [float(raw)]
        return out

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[list, dict[str, Any]]:
        return [0.0] * self._obs_dim, {}

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        return [0.0] * self._obs_dim, 0.5, False, {}

    def render(self, mode: str = "human") -> None:
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class TestStructuredObsGate:
    """D1: the structured (Dict) obs space is opt-in; metrics-only stays flat Box."""

    def test_metrics_only_env_falls_back_to_box(self) -> None:
        """An env that opts out (``structured_observation_descriptors`` -> None) keeps a flat Box.

        This is the blast-radius guard: non-DR workloads (BO/GA/MAB on plain
        metrics) must NOT silently switch to a Dict obs layout.
        """
        import gymnasium

        gym_env = _StructuredStubBaseGym(descriptors=None, obs_dim=3)
        adapter = GymnasiumAdapter(gym_env)

        assert isinstance(adapter.observation_space, gymnasium.spaces.Box)
        assert adapter.observation_space.shape == (3,)

    def test_env_param_env_uses_dict(self) -> None:
        """An env with a declared env_param leaf exposes a ``spaces.Dict``."""
        import gymnasium

        descriptors = {
            "bus_bw": ObsLeafDescriptor(kind="box", dim=1),
            "drop_rate": ObsLeafDescriptor(kind="box", dim=2),
        }
        adapter = GymnasiumAdapter(_StructuredStubBaseGym(descriptors=descriptors))

        assert isinstance(adapter.observation_space, gymnasium.spaces.Dict)
        assert set(adapter.observation_space.spaces) == {"bus_bw", "drop_rate"}
        assert adapter.observation_space.spaces["drop_rate"].shape == (2,)


class TestCategoricalLeafSubspace:
    """D3: a categorical (discrete) descriptor maps to ``Discrete(k)`` and decodes to an int."""

    def test_discrete_descriptor_becomes_discrete_space(self) -> None:
        import gymnasium

        descriptors = {
            "bus_bw": ObsLeafDescriptor(kind="box", dim=1),
            "variant": ObsLeafDescriptor(kind="discrete", dim=1, n=3),
        }
        adapter = GymnasiumAdapter(_StructuredStubBaseGym(descriptors=descriptors))

        observation_space = adapter.observation_space
        assert isinstance(observation_space, gymnasium.spaces.Dict)
        variant_space = observation_space.spaces["variant"]
        assert isinstance(variant_space, gymnasium.spaces.Discrete)
        assert int(variant_space.n) == 3

    def test_discrete_leaf_emitted_as_int_index(self) -> None:
        """The emitted obs for a discrete leaf is an ``int`` the Discrete space accepts."""
        descriptors = {"variant": ObsLeafDescriptor(kind="discrete", dim=1, n=3)}
        gym_env = _StructuredStubBaseGym(descriptors=descriptors, obs_dim=1)
        adapter = GymnasiumAdapter(gym_env)

        obs, _ = adapter.reset()
        assert isinstance(obs["variant"], int)
        assert adapter.observation_space.contains(obs)
