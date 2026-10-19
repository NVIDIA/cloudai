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

"""Configuration models for environment-parameter sampling and observations."""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ObsLeafDescriptor(BaseModel):
    """
    Shape of one leaf of a structured (named) observation.

    Adapters use it to build the matching subspace without guessing: a ``"box"`` leaf is a
    continuous vector of width ``dim``; a ``"discrete"`` leaf is a categorical of size ``n``.
    Agents that consume the flat observation ignore it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["box", "discrete"]
    dim: int = 1
    n: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.dim < 1:
            raise ValueError(f"ObsLeafDescriptor dim must be >= 1; got {self.dim}")
        if self.kind == "discrete" and (self.n is None or self.n < 1):
            raise ValueError(f"ObsLeafDescriptor(kind='discrete') requires n >= 1; got n={self.n}")
        return self


class CategoricalEncoding(BaseModel):
    """
    Default encoding: observe the drawn value as its categorical index into ``candidates``.

    Candidates are a discrete set (a ``cmd_args`` list), so the policy sees the per-trial
    regime as a ``Discrete(len(candidates))`` index rather than a raw magnitude: an
    arbitrary candidate list carries no ordinal meaning to encode continuously.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical"] = "categorical"

    def observation_descriptor(self, candidates: List[Any]) -> ObsLeafDescriptor:
        return ObsLeafDescriptor(kind="discrete", n=len(candidates))

    def encode(self, value: Any, candidates: List[Any]) -> int:
        return candidates.index(value)


class EnvParamSpec(BaseModel):
    """
    Annotation marking one cmd_args field as env-sampled.

    Carries only *how* to sample - the candidate values themselves live in
    ``cmd_args.<name>`` as a plain list. ``weights`` (optional) are positional,
    aligned 1:1 with that candidate list; omit for uniform sampling. ``encoding``
    (optional) selects how the drawn value is exposed to the policy as an
    observation leaf, defaulting to a categorical index. The length match against
    the candidate list is a cross-field check enforced by ``TestDefinition`` (which
    can see ``cmd_args``); here we validate only the weights' intrinsic shape.
    """

    model_config = ConfigDict(extra="forbid")

    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional probability weights aligned with the cmd_args candidate list; uniform if omitted.",
    )
    encoding: CategoricalEncoding = Field(
        default_factory=CategoricalEncoding,
        description="How the drawn value is encoded as an observation leaf (categorical index into the candidates).",
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        if self.weights is None:
            return self
        for w in self.weights:
            if not math.isfinite(w) or w < 0:
                raise ValueError(f"env_params weights must be finite and non-negative; got {w}")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"env_params weights must sum to 1.0; got {total}")
        return self
