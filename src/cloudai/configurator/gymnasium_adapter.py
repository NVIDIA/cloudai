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

from typing import Any, ClassVar

from .base_gym import BaseGym


def _import_gymnasium():
    """Import gymnasium lazily; raise a clear error when it is absent."""
    try:
        import gymnasium
        from gymnasium import spaces

        return gymnasium, spaces
    except ImportError as exc:
        raise ImportError("gymnasium is required for GymnasiumAdapter. Install it with: pip install gymnasium") from exc


class GymnasiumAdapter:
    """
    Wrap a CloudAI BaseGym environment as a standard gymnasium.Env.

    gymnasium is imported lazily so it remains an optional dependency.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}

    def __init__(self, env: BaseGym) -> None:
        import numpy as np

        gymnasium, spaces = _import_gymnasium()

        gymnasium.Env.__init__(self)

        self._np = np
        self._env = env

        raw_action_space = env.define_action_space()
        self._tunable_params: dict[str, list] = {k: v for k, v in raw_action_space.items() if len(v) > 1}

        self.action_space = spaces.Dict({k: spaces.Discrete(len(v)) for k, v in self._tunable_params.items()})

        obs = env.define_observation_space()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return (observation, info)."""
        obs, info = self._env.reset(seed=seed, options=options)
        return self._np.asarray(obs, dtype=self._np.float32), info

    def step(self, action: dict[str, int]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one step and return the gymnasium 5-tuple."""
        decoded = self.decode_action(action)
        obs, reward, done, info = self._env.step(decoded)
        return self._np.asarray(obs, dtype=self._np.float32), float(reward), bool(done), False, info

    def decode_action(self, action: dict[str, int]) -> dict[str, Any]:
        """Map discrete indices back to the original parameter values."""
        return {k: self._tunable_params[k][idx] for k, idx in action.items()}

    def render(self) -> None:
        """Render the underlying environment."""
        self._env.render()

    @property
    def unwrapped(self) -> BaseGym:
        """Return the wrapped CloudAI BaseGym instance."""
        return self._env
