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
producing duplicate-step trajectory records.

These tests pin the negative invariant: the adapter must not mutate
``test_run.step``. That counter is owned by ``TestRun`` and advanced
exclusively by ``CloudAIGymEnv.step()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

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


class TestEncodeDecodeAreInverse:
    """``encode_action`` is the inverse of ``decode_action`` on native values.

    Consumers (e.g. RLlib warm-start / behavioral cloning) must be able to
    express a recorded native config in the policy's action space without
    reaching into adapter internals. The public pair guarantees
    ``decode_action(encode_action(v)) == v`` for any native ``v``.
    """

    def test_discrete_round_trip_decode_of_encode_is_identity(self) -> None:
        adapter = GymnasiumAdapter(_StubBaseGym())
        native = {"param_a": 3, "param_b": 10}
        assert adapter.decode_action(adapter.encode_action(native)) == native

    def test_discrete_encode_of_decode_is_identity(self) -> None:
        adapter = GymnasiumAdapter(_StubBaseGym())
        action = {"param_a": 2, "param_b": 1}
        assert adapter.encode_action(adapter.decode_action(action)) == action

    def test_encode_maps_value_to_candidate_index(self) -> None:
        adapter = GymnasiumAdapter(_StubBaseGym())
        assert adapter.encode_action({"param_a": 2, "param_b": 20}) == {"param_a": 1, "param_b": 1}

    def test_encode_rejects_non_candidate_value(self) -> None:
        adapter = GymnasiumAdapter(_StubBaseGym())
        with pytest.raises(ValueError, match="not a candidate"):
            adapter.encode_action({"param_a": 7, "param_b": 10})

    def test_encode_rejects_key_mismatch(self) -> None:
        adapter = GymnasiumAdapter(_StubBaseGym())
        with pytest.raises(ValueError, match="keys mismatch"):
            adapter.encode_action({"param_a": 1})  # missing param_b


class _StructuredStubBaseGym(BaseGym):
    """BaseGym stub that opts in/out of the structured (Dict) obs path.

    Mirrors ``CloudAIGymEnv``'s contract: ``structured_observation_descriptors``
    returns ``None`` for a metrics-only env and a per-leaf descriptor dict otherwise;
    the per-trial regime is reported on ``info["env_params"]``; ``encode_env_params``
    turns that regime into the matching encoded leaves.
    """

    def __init__(
        self,
        descriptors: Optional[dict[str, ObsLeafDescriptor]],
        obs_dim: int = 2,
        regime: Optional[dict[str, Any]] = None,
    ) -> None:
        self._descriptors = descriptors
        self._obs_dim = obs_dim
        if regime is not None:
            self._regime = regime
        elif descriptors:
            self._regime = {name: (0 if d.kind == "discrete" else 0.0) for name, d in descriptors.items()}
        else:
            self._regime = {}
        self._action_space: dict[str, Any] = {"param_a": [1, 2, 3]}
        self.test_run = SimpleNamespace(step=0)
        super().__init__()

    def define_action_space(self) -> dict[str, Any]:
        return self._action_space

    def define_observation_space(self) -> list:
        return [0.0] * self._obs_dim

    def structured_observation_descriptors(self) -> Optional[dict[str, ObsLeafDescriptor]]:
        return self._descriptors

    def encode_env_params(self, env_params: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        assert self._descriptors is not None
        for name, desc in self._descriptors.items():
            raw = env_params[name]
            if desc.kind == "discrete":
                out[name] = int(raw)
            elif desc.dim == 2:
                out[name] = [1.0 if raw <= 0.0 else 0.0, float(raw)]
            else:
                out[name] = [float(raw)]
        return out

    def _info(self) -> dict[str, Any]:
        return {"env_params": dict(self._regime)} if self._descriptors is not None else {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[list, dict[str, Any]]:
        return [0.0] * self._obs_dim, self._info()

    def step(self, action: Any) -> tuple[list, float, bool, dict]:
        return [0.0] * self._obs_dim, 0.5, False, self._info()

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
        """An env with declared env_params exposes ``Dict({"observation": Box, "context": Dict(...)})``."""
        import gymnasium

        descriptors = {
            "bus_bw": ObsLeafDescriptor(kind="box", dim=1),
            "drop_rate": ObsLeafDescriptor(kind="box", dim=2),
        }
        adapter = GymnasiumAdapter(_StructuredStubBaseGym(descriptors=descriptors))

        observation_space = adapter.observation_space
        assert isinstance(observation_space, gymnasium.spaces.Dict)
        assert set(observation_space) == {"observation", "context"}
        context_space = observation_space["context"]
        assert isinstance(context_space, gymnasium.spaces.Dict)
        assert set(context_space) == {"bus_bw", "drop_rate"}
        assert context_space["drop_rate"].shape == (2,)


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
        context_space = observation_space["context"]
        assert isinstance(context_space, gymnasium.spaces.Dict)
        variant_space = context_space["variant"]
        assert isinstance(variant_space, gymnasium.spaces.Discrete)
        assert int(variant_space.n) == 3

    def test_discrete_leaf_emitted_as_int_index(self) -> None:
        """The emitted obs for a discrete leaf is an ``int`` the Discrete space accepts."""
        descriptors = {"variant": ObsLeafDescriptor(kind="discrete", dim=1, n=3)}
        gym_env = _StructuredStubBaseGym(descriptors=descriptors, obs_dim=1, regime={"variant": 2})
        adapter = GymnasiumAdapter(gym_env)

        obs, _ = adapter.reset()
        assert isinstance(obs["context"]["variant"], int)
        assert adapter.observation_space.contains(obs)
