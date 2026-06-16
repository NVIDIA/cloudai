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
Gymnasium adapter for CloudAI ``BaseGym`` environments.

Translates a CloudAI :class:`BaseGym` into the ``gymnasium.Env`` 5-tuple shape
that RLlib-based agents (e.g. PPO / DQN) and external training loops expect.
``gymnasium`` is an optional dependency (the ``[rl]`` extra), so it is imported
lazily and only required when an adapter is actually instantiated.

Design invariant — adapter is a pure pass-through over ``test_run.step``.
The trial counter is owned by ``TestRun`` and advanced exclusively by
``CloudAIGymEnv.step()``. Adapters that wrote ``test_run.step`` themselves —
mirroring a Gym-protocol episode-local counter — collapsed every
contextual-bandit rollout onto ``step=1`` because RLlib calls ``reset()`` per
trial. This adapter never mutates ``test_run.step``; contract tests pin that
property.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, cast

from cloudai._core.action_space import ContinuousSpace
from cloudai.configurator.base_gym import BaseGym
from cloudai.configurator.env_params import StructuredObservation

_GYMNASIUM_INSTALL_HINT = "gymnasium is required for GymnasiumAdapter. Install it with: pip install 'cloudai[rl]'"


def _import_gymnasium():
    """
    Import gymnasium + numpy lazily; raise a clear, actionable error when absent.

    Kept as a single seam so cloudai installs without the ``[rl]`` extra
    continue to work for non-RL agents, and tests can patch this helper to
    simulate a missing install.
    """
    try:
        import gymnasium
        import numpy as np
        from gymnasium import spaces

        return gymnasium, spaces, np
    except ImportError as exc:
        raise ImportError(_GYMNASIUM_INSTALL_HINT) from exc


class GymnasiumAdapter:
    """
    Expose a CloudAI :class:`BaseGym` as a ``gymnasium.Env``-shaped object.

    The adapter:

    * Builds a ``gymnasium.spaces.Dict`` of ``Discrete`` action spaces over
      the *tunable* parameters (those with more than one candidate value),
      and injects the *fixed* parameters (single candidate) automatically on
      every step so agents never see them.
    * Converts observations to ``float32`` ``numpy`` arrays sized by
      ``env.define_observation_space()``.
    * Returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
      from :meth:`step` and :meth:`step_raw`.

    ``gymnasium`` and ``numpy`` are optional dependencies (the ``[rl]`` extra);
    instantiating the adapter without them raises ``ImportError``.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}

    def __init__(self, env: BaseGym) -> None:
        _, spaces, np = _import_gymnasium()

        self._np = np
        self._spaces = spaces
        self._env = env

        raw_action_space = env.define_action_space()

        # Three classes of params from cloudai's param_space:
        #   list with len > 1   -> discrete tunable, mapped to gym.Discrete.
        #   list with len == 1  -> fixed (collapsed); injected on every step so
        #                          agents never see them.
        #   ContinuousSpace     -> continuous tunable, mapped to gym.Box(1,);
        #                          ``dtype="int"`` quantizes at decode_action.
        self._discrete_params: dict[str, list] = {
            k: v for k, v in raw_action_space.items() if isinstance(v, list) and len(v) > 1
        }
        self._continuous_params: dict[str, ContinuousSpace] = {
            k: v for k, v in raw_action_space.items() if isinstance(v, ContinuousSpace)
        }
        self._fixed_params: dict[str, Any] = {
            k: v[0] for k, v in raw_action_space.items() if isinstance(v, list) and len(v) == 1
        }

        action_space_components: dict[str, Any] = {
            name: spaces.Discrete(len(values)) for name, values in self._discrete_params.items()
        }
        for name, space in self._continuous_params.items():
            action_space_components[name] = spaces.Box(
                low=np.array([space.low], dtype=np.float32),
                high=np.array([space.high], dtype=np.float32),
                shape=(1,),
                dtype=np.float32,
            )
        self.action_space = spaces.Dict(action_space_components)

        # Observation space: prefer the env's structured (per-leaf) spec so the
        # policy sees named, individually-encoded leaves (e.g. a log-encoded
        # env_param as Box(2)); RLlib connectors own normalize + flatten. Falls
        # back to a flat Box for envs that only expose define_observation_space.
        self._obs_descriptors: Optional[dict[str, Any]] = self._structured_obs_descriptors(env)
        if self._obs_descriptors:
            self.observation_space = spaces.Dict(
                {name: self._descriptor_to_space(desc) for name, desc in self._obs_descriptors.items()}
            )
        else:
            obs_shape = (len(env.define_observation_space()),)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    @staticmethod
    def _structured_obs_descriptors(env: BaseGym) -> Optional[dict[str, Any]]:
        """
        Return the env's per-leaf obs descriptors, or ``None`` for the flat-Box path.

        The env owns the opt-in decision via ``structured_observation_descriptors``
        (returns ``None`` unless an observed name is a declared env_param). Envs
        without that hook keep the legacy flat-Box path.
        """
        getter = getattr(env, "structured_observation_descriptors", None)
        if getter is None or not hasattr(env, "encode_observation"):
            return None
        descriptors = getter()
        return descriptors or None

    def _descriptor_to_space(self, descriptor: Any) -> Any:
        """Map a framework-agnostic ``ObsLeafDescriptor`` to a gymnasium subspace."""
        if descriptor.kind == "discrete":
            return self._spaces.Discrete(descriptor.n)
        return self._spaces.Box(low=-self._np.inf, high=self._np.inf, shape=(descriptor.dim,), dtype=self._np.float32)

    @property
    def unwrapped(self) -> BaseGym:
        return self._env

    def decode_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Map raw gym actions back to native parameter values.

        Discrete actions are list indices and resolve to the corresponding list
        entry. Continuous actions arrive as a 1-D ``numpy`` array (the Box
        shape) and are clamped to the declared range; for ``dtype="int"`` the
        scalar is rounded — this is the **only** place quantization happens, so
        the cache key collapses float jitter (47.34 vs 47.36) onto the same
        emitted int and downstream code stays type-agnostic.

        Raises:
            ValueError: if ``action`` is missing tunable params, contains
                unknown keys, or carries an out-of-range discrete index.
        """
        expected = set(self._discrete_params) | set(self._continuous_params)
        self._assert_keys(action.keys(), expected, "action")
        decoded: dict[str, Any] = {}
        for name, raw in action.items():
            if name in self._discrete_params:
                decoded[name] = self._decode_discrete(name, raw)
            else:
                decoded[name] = self._decode_continuous(name, raw)
        return decoded

    def _decode_discrete(self, name: str, raw: Any) -> Any:
        values = self._discrete_params[name]
        idx = int(raw)
        if not 0 <= idx < len(values):
            raise ValueError(f"Action index out of range for '{name}': {idx} (expected 0..{len(values) - 1})")
        return values[idx]

    def _decode_continuous(self, name: str, raw: Any) -> Any:
        space = self._continuous_params[name]
        scalar = float(self._np.asarray(raw, dtype=self._np.float32).reshape(-1)[0])
        clamped = min(max(scalar, space.low), space.high)
        return round(clamped) if space.dtype == "int" else clamped

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return self._as_obs_array(obs), info

    def step(self, action: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        params = {**self._fixed_params, **self.decode_action(action)}
        return self._step_with_params(params)

    def step_raw(self, params: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Step the env with an already-decoded parameter dict; bypasses index decoding.

        Raises:
            ValueError: if ``params`` does not cover exactly the tunable +
                fixed param keys.
        """
        expected = set(self._discrete_params) | set(self._continuous_params) | set(self._fixed_params)
        self._assert_keys(params.keys(), expected, "raw params")
        return self._step_with_params(params)

    def render(self) -> None:
        self._env.render()

    @staticmethod
    def _assert_keys(received: Any, expected: set[str], ctx: str) -> None:
        received_set = set(received)
        if received_set == expected:
            return
        missing = sorted(expected - received_set)
        extra = sorted(received_set - expected)
        raise ValueError(f"{ctx} keys mismatch; missing={missing}, extra={extra}")

    def _step_with_params(self, params: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(params)
        return self._as_obs_array(obs), float(reward), bool(done), False, info

    def _as_obs_array(self, obs: Any) -> Any:
        """
        Convert a raw env observation into the policy-facing observation.

        Structured path: the flat raw obs (which feeds the env's reward and
        ``trajectory.csv``) is encoded per-leaf by the env and returned as a
        ``dict`` keyed to match ``observation_space`` (a ``spaces.Dict``).
        Flat path: a single ``float32`` ``Box`` array (legacy behaviour).
        """
        descriptors = self._obs_descriptors
        if descriptors is None:
            return self._np.asarray(obs, dtype=self._np.float32)
        env = cast(StructuredObservation, self._env)
        encoded = env.encode_observation(list(obs))
        return {name: self._leaf_to_value(descriptors[name], leaf) for name, leaf in encoded.items()}

    def _leaf_to_value(self, descriptor: Any, leaf: Any) -> Any:
        """Coerce one encoded leaf to its gymnasium subspace dtype."""
        if descriptor.kind == "discrete":
            return int(leaf)
        return self._np.asarray(leaf, dtype=self._np.float32)
