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

from typing import TYPE_CHECKING, Any, Optional

from cloudai.util.lazy_imports import lazy

from .base_gym import BaseGym
from .env_params import StructuredObservationProducer

if TYPE_CHECKING:
    from gymnasium import Env as _GymnasiumEnvBase
else:
    try:  # ``gymnasium`` is an optional [rl] dependency; fall back to ``object`` when absent.
        from gymnasium import Env as _GymnasiumEnvBase
    except ImportError:
        _GymnasiumEnvBase = object


class GymnasiumAdapter(_GymnasiumEnvBase):
    """
    Expose a CloudAI :class:`BaseGym` as a ``gymnasium.Env``-shaped object.

    The adapter:

    * Builds a ``gymnasium.spaces.Dict`` of ``Discrete`` action spaces over
      the *tunable* parameters (those with more than one candidate value),
      and injects the *fixed* parameters (single candidate) automatically on
      every step so agents never see them.
    * Builds the observation from the env's flat metrics: a ``float32`` ``Box`` for
      metrics-only envs, or — when the env declares env_params — a ``spaces.Dict``
      ``{"observation": Box, "context": Dict(...)}`` pairing the metrics ``observation``
      with the per-trial ``context`` the env reports on ``info["env_params"]``, each
      param encoded to its leaf. Naming follows the contextual-MDP convention
      (``observation`` = inner signal, ``context`` = env regime).
    * Returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
      from :meth:`step` and :meth:`step_raw`.

    ``gymnasium`` and ``numpy`` are optional dependencies (the ``[rl]`` extra);
    instantiating the adapter without them raises ``ImportError``.
    """

    # Overrides gymnasium.Env.metadata (a non-ClassVar instance attribute); matching that
    # shape satisfies pyright's override check, so RUF012's ClassVar suggestion is silenced.
    metadata: dict[str, Any] = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, env: BaseGym) -> None:
        np = self._np = lazy.np
        spaces = self._spaces = lazy.gymnasium.spaces
        self._env = env

        raw_action_space = env.define_action_space()

        # Two classes of params from cloudai's param_space:
        #   list with len > 1   -> discrete tunable, mapped to gym.Discrete.
        #   list with len == 1  -> fixed (collapsed); injected on every step so
        #                          agents never see them.
        self._discrete_params: dict[str, list] = {
            k: v for k, v in raw_action_space.items() if isinstance(v, list) and len(v) > 1
        }
        self._fixed_params: dict[str, Any] = {
            k: v[0] for k, v in raw_action_space.items() if isinstance(v, list) and len(v) == 1
        }

        action_space_components: dict[str, Any] = {
            name: spaces.Discrete(len(values)) for name, values in self._discrete_params.items()
        }
        self.action_space = spaces.Dict(action_space_components)

        # Observation space. When the env declares env_params, expose a structured
        # spaces.Dict pairing the flat metrics (unchanged) with the per-trial context:
        #   {"observation": Box(m), "context": Dict({<param>: <leaf subspace>, ...})}
        # so the policy sees named, individually-encoded context leaves (e.g. a log-encoded
        # env_param as Box(2)) alongside the metrics; RLlib connectors own normalize + flatten.
        # Envs without env_params keep the legacy flat Box.
        metrics_shape = (len(env.define_observation_space()),)
        metrics_space = spaces.Box(low=-np.inf, high=np.inf, shape=metrics_shape, dtype=np.float32)
        self._obs_descriptors: Optional[dict[str, Any]] = self._structured_obs_descriptors(env)
        if self._obs_descriptors:
            context_space = spaces.Dict(
                {name: self._descriptor_to_space(desc) for name, desc in self._obs_descriptors.items()}
            )
            self.observation_space = spaces.Dict({"observation": metrics_space, "context": context_space})
        else:
            self.observation_space = metrics_space

    @staticmethod
    def _structured_obs_descriptors(env: BaseGym) -> Optional[dict[str, Any]]:
        """
        Return the env's per-leaf obs descriptors, or ``None`` for the flat-Box path.

        The env owns the opt-in decision via ``structured_observation_descriptors``
        (returns ``None`` unless it declares env_params). Envs that don't satisfy the
        ``StructuredObservationProducer`` protocol keep the legacy flat-Box path.
        """
        if not isinstance(env, StructuredObservationProducer):
            return None
        descriptors = env.structured_observation_descriptors()
        return descriptors or None

    def _descriptor_to_space(self, descriptor: Any) -> Any:
        """Map a framework-agnostic ``ObsLeafDescriptor`` to a gymnasium subspace."""
        if descriptor.kind == "discrete":
            return self._spaces.Discrete(descriptor.n)
        return self._spaces.Box(low=-self._np.inf, high=self._np.inf, shape=(descriptor.dim,), dtype=self._np.float32)

    def decode_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Map raw gym actions back to native parameter values.

        Discrete actions are list indices and resolve to the corresponding list
        entry.

        Raises:
            ValueError: if ``action`` is missing tunable params, contains
                unknown keys, or carries an out-of-range discrete index.
        """
        self._assert_keys(action.keys(), set(self._discrete_params), "action")
        return {name: self._decode_discrete(name, raw) for name, raw in action.items()}

    def _decode_discrete(self, name: str, raw: Any) -> Any:
        values = self._discrete_params[name]
        idx = int(raw)
        if not 0 <= idx < len(values):
            raise ValueError(f"Action index out of range for '{name}': {idx} (expected 0..{len(values) - 1})")
        return values[idx]

    def encode_action(self, values: dict[str, Any]) -> dict[str, Any]:
        """
        Map native parameter values back to raw gym actions; inverse of :meth:`decode_action`.

        Discrete values resolve to their index in the candidate list. Together
        with :meth:`decode_action` this is an invertible pair on native values:
        ``decode_action(encode_action(v)) == v`` for any ``v`` drawn from the
        tunable params.

        Consumers that need to express known native configs in the policy's
        action space — e.g. warm-start / behavioral cloning from a recorded
        trajectory — call this instead of reaching into the adapter internals.

        Raises:
            ValueError: if ``values`` does not cover exactly the tunable params,
                or carries a discrete value absent from its candidate list.
        """
        self._assert_keys(values.keys(), set(self._discrete_params), "values")
        return {name: self._encode_discrete(name, value) for name, value in values.items()}

    def _encode_discrete(self, name: str, value: Any) -> int:
        values = self._discrete_params[name]
        try:
            return values.index(value)
        except ValueError:
            raise ValueError(f"Value {value!r} for '{name}' is not a candidate; expected one of {values}") from None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return self._as_obs(obs, info), info

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
        expected = set(self._discrete_params) | set(self._fixed_params)
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
        return self._as_obs(obs, info), float(reward), bool(done), False, info

    def _as_obs(self, obs: Any, info: dict[str, Any]) -> Any:
        """
        Build the policy-facing observation from the env's flat obs and its ``info``.

        Flat path: a single ``float32`` ``Box`` array (the metrics), legacy behaviour.
        Structured path: a ``spaces.Dict`` pairing the flat metrics with the per-trial
        env_param regime the env reports on ``info["env_params"]``, each param encoded to
        its declared leaf: ``{"observation": Box(m), "context": {<param>: <leaf>, ...}}``.
        """
        metrics = self._np.asarray(obs, dtype=self._np.float32)
        descriptors = self._obs_descriptors
        env = self._env
        if descriptors is None or not isinstance(env, StructuredObservationProducer):
            return metrics
        encoded = env.encode_env_params(info["env_params"])
        self._assert_keys(encoded.keys(), set(descriptors), "encoded env_params")
        context_leaves = {name: self._leaf_to_value(descriptors[name], encoded[name]) for name in descriptors}
        return {"observation": metrics, "context": context_leaves}

    def _leaf_to_value(self, descriptor: Any, leaf: Any) -> Any:
        """Coerce one encoded leaf to its gymnasium subspace dtype."""
        if descriptor.kind == "discrete":
            return int(leaf)
        return self._np.asarray(leaf, dtype=self._np.float32)
