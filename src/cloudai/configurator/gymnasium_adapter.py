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

from typing import Any, ClassVar, Optional

from .base_gym import BaseGym

_GYMNASIUM_INSTALL_HINT = "gymnasium is required for GymnasiumAdapter. Install it with: pip install gymnasium"


def _import_gymnasium():
    """
    Import gymnasium + numpy lazily; raise a clear, actionable error when absent.

    Kept as a single seam so that:
    * cloudai installs without ``gymnasium`` continue to work for users that don't
      need this adapter (the import is gated behind ``GymnasiumAdapter()``);
    * tests can patch this helper to simulate a missing install.
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
    Expose a CloudAI :class:`BaseGym` as a standard ``gymnasium.Env``-shaped object.

    The adapter:

    * builds a ``gymnasium.spaces.Dict`` of ``Discrete`` action spaces over the
      *tunable* parameters (those with more than one candidate value), and
      injects the *fixed* parameters (single candidate) automatically on every
      step so agents never see them.
    * converts observations to ``float32`` ``numpy`` arrays sized by
      ``env.define_observation_space()``.
    * returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
      from :meth:`step` and :meth:`step_raw`.
    * keeps ``env.test_run.step`` in sync (1-based) so artifact paths produced by
      ``CloudAIGymEnv`` match those produced by ``handle_dse_job`` (i.e.
      ``<scenario>/<test>/<iteration>/<step>/`` for every evaluation), which is
      required when a custom training loop (e.g. RLlib) front-ends the env.

    ``gymnasium`` and ``numpy`` are optional dependencies; importing this module
    is cheap, but instantiating the adapter without them raises ``ImportError``.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}

    def __init__(self, env: BaseGym) -> None:
        _, spaces, np = _import_gymnasium()

        self._np = np
        self._env = env
        self._step_count = 0

        raw_action_space = env.define_action_space()
        self._tunable_params: dict[str, list] = {k: v for k, v in raw_action_space.items() if len(v) > 1}
        self._fixed_params: dict[str, Any] = {k: v[0] for k, v in raw_action_space.items() if len(v) == 1}

        self.action_space = spaces.Dict(
            {name: spaces.Discrete(len(values)) for name, values in self._tunable_params.items()}
        )

        obs_shape = (len(env.define_observation_space()),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    @property
    def unwrapped(self) -> BaseGym:
        return self._env

    def decode_action(self, action: dict[str, int]) -> dict[str, Any]:
        """Map discrete action indices back to the original parameter values."""
        return {name: self._tunable_params[name][idx] for name, idx in action.items()}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        obs, info = self._env.reset(seed=seed, options=options)
        return self._as_obs_array(obs), info

    def step(self, action: dict[str, int]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        params = {**self._fixed_params, **self.decode_action(action)}
        return self._step_with_params(params)

    def step_raw(self, params: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the env with an already-decoded parameter dict; bypasses index decoding."""
        return self._step_with_params(params)

    def render(self) -> None:
        self._env.render()

    def _step_with_params(self, params: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._sync_underlying_step_counter()
        obs, reward, done, info = self._env.step(params)
        self._step_count += 1
        return self._as_obs_array(obs), float(reward), bool(done), False, info

    def _sync_underlying_step_counter(self) -> None:
        """
        Mirror ``handle_dse_job``'s 1-based ``test_run.step`` so artifact paths match.

        The first step is written under ``…/<iteration>/1/``, matching how
        ``handle_dse_job`` numbers steps; this keeps reports and trajectory
        analysis consistent regardless of whether the env is driven by the
        DSE loop or by an external training loop wrapping the adapter.
        """
        test_run = getattr(self._env, "test_run", None)
        if test_run is not None:
            test_run.step = self._step_count + 1

    def _as_obs_array(self, obs: Any) -> Any:
        return self._np.asarray(obs, dtype=self._np.float32)
