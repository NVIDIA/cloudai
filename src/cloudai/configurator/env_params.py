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
Per-trial environment-parameter primitives for CloudAI DSE.

An env-randomized parameter is a workload knob whose candidate values live in
``cmd_args`` (a plain list - the single source of truth, exactly like an
action-space dimension) but which the environment *samples* per trial rather
than letting the agent search it. ``env_params`` is the annotation that marks
such a field; it carries only *how* to sample (optional ``weights``), never the
values, and the knob never enters the agent's action space.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from cloudai._core.exceptions import TestScenarioParsingError

if TYPE_CHECKING:
    from cloudai._core.test_scenario import TestScenario
    from cloudai.models.workload import TestDefinition


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


class Encoding(Protocol):
    """
    Strategy mapping an env_param's drawn value to an observation leaf.

    An encoding declares its own :class:`ObsLeafDescriptor` and encodes a drawn value
    into a leaf of that shape. A new strategy implements this pair without touching
    :class:`EnvParam` or the adapter.
    """

    def observation_descriptor(self, candidates: List[Any]) -> ObsLeafDescriptor: ...

    def encode(self, value: Any, candidates: List[Any]) -> Any: ...


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


class LogEncoding(BaseModel):
    """
    Log-scale encoding: observe the drawn value as its log-scaled float.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["log"] = "log"

    def observation_descriptor(self, candidates: List[Any]) -> ObsLeafDescriptor:
        return ObsLeafDescriptor(kind="box", dim=1)

    def encode(self, value: Any, candidates: List[Any]) -> float:
        return float(math.log(float(value)))


AnyEncoding = Annotated[
    Union[CategoricalEncoding, LogEncoding],
    Field(discriminator="type")
]


def _infer_encoding(candidates: List[Any]) -> AnyEncoding:
    """Infer the appropriate encoding strategy for a list of candidate values."""
    if not candidates:
        return CategoricalEncoding()

    if all(isinstance(c, str) for c in candidates):
        return CategoricalEncoding()

    if len(candidates) < 3:
        return CategoricalEncoding()

    if any(isinstance(c, bool) for c in candidates):
        return CategoricalEncoding()

    if not all(isinstance(c, (int, float)) and c > 0 for c in candidates):
        return CategoricalEncoding()

    try:
        sorted_c = sorted(float(c) for c in candidates)
    except (ValueError, TypeError):
        return CategoricalEncoding()

    # Check perfectly uniform diffs (arithmetic) -> not log
    diffs = [sorted_c[i] - sorted_c[i-1] for i in range(1, len(sorted_c))]
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff > 0 and all(math.isclose(d, avg_diff, rel_tol=1e-5) for d in diffs):
        return CategoricalEncoding()

    # Check constant ratio within tolerance (geometric series)
    ratios = [sorted_c[i] / sorted_c[i-1] for i in range(1, len(sorted_c))]
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio > 1.0 + 1e-9 and all(math.isclose(r, avg_ratio, rel_tol=1e-5) for r in ratios):
        return LogEncoding()

    return CategoricalEncoding()


class EnvParamSpec(BaseModel):
    """
    Annotation marking one cmd_args field as env-sampled.

    Carries only *how* to sample - the candidate values themselves live in
    ``cmd_args.<name>`` as a plain list. ``weights`` (optional) are positional,
    aligned 1:1 with that candidate list; omit for uniform sampling. ``encoding``
    (optional) selects how the drawn value is exposed to the policy as an
    observation leaf, defaulting to None and is inferred from candidates. The length match against
    the candidate list is a cross-field check enforced by ``TestDefinition`` (which
    can see ``cmd_args``); here we validate only the weights' intrinsic shape.
    """

    model_config = ConfigDict(extra="forbid")

    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional probability weights aligned with the cmd_args candidate list; uniform if omitted.",
    )
    encoding: Optional[AnyEncoding] = Field(
        default=None,
        description="How the drawn value is encoded as an observation leaf. If omitted, inferred from candidates.",
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


@runtime_checkable
class StructuredObservationProducer(Protocol):
    """
    Optional env hooks exposing the env_params behind the (unchanged, flat) observation.

    The env keeps returning its flat observation (the metrics) and, when a regime was applied,
    delivers it on the Gym ``info`` dict under ``info["env_params"]`` (the key is absent otherwise,
    so its presence alone signals a non-empty regime). An env opts in by
    declaring per-leaf :class:`ObsLeafDescriptor` for its env_params via
    ``structured_observation_descriptors`` (``None`` when none are declared) and encoding a regime
    into the matching named leaves via ``encode_env_params``. ``GymnasiumAdapter`` merges the flat
    metrics with these env_param leaves into its structured observation ``gymnasium.spaces.Dict``.
    The hooks are duck-typed, so envs need not subclass this Protocol.
    """

    def structured_observation_descriptors(self) -> Optional[Dict[str, ObsLeafDescriptor]]: ...

    def encode_env_params(self, env_params: dict[str, Any]) -> Dict[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class EnvParam:
    """
    One env-sampled knob, resolved from cmd_args: candidates, optional weights, and encoding.

    Weights (when present) are positional, aligned 1:1 with ``candidates``; ``None`` means
    uniform sampling. ``encoding`` owns how a drawn value is exposed to the policy as an
    observation leaf. Bundling the three makes each knob a self-contained draw-and-encode.
    """

    candidates: List[Any]
    weights: Optional[List[float]] = None
    encoding: Encoding = dataclasses.field(default_factory=CategoricalEncoding)

    def draw(self, rng: random.Random) -> Any:
        if self.weights is not None:
            return rng.choices(self.candidates, weights=self.weights, k=1)[0]
        return rng.choice(self.candidates)

    def observation_descriptor(self) -> ObsLeafDescriptor:
        return self.encoding.observation_descriptor(self.candidates)

    def encode(self, value: Any) -> Any:
        return self.encoding.encode(value, self.candidates)


@dataclasses.dataclass(frozen=True)
class EnvParams:
    """
    Resolved env-parameter sampling state for one run.

    Built via ``from_test`` only when a workload actually declares list-valued env_params;
    otherwise ``from_test`` returns ``None`` and the env carries no env-params state at all.
    Owns the per-parameter :class:`EnvParam` draws (resolved from ``cmd_args``, the single source
    of truth) and the seed, and draws one value per parameter per trial.
    """

    params: Dict[str, EnvParam]
    seed: int

    @classmethod
    def from_test(cls, test: "TestDefinition") -> Optional["EnvParams"]:
        """
        Resolve a TestDefinition's env_params annotations, or ``None`` if nothing is sampled.

        Annotated fields are guaranteed list-valued by ``TestDefinition`` parse-time validation
        (a scalar annotation is rejected there), so the non-list guard below is defensive. With
        no annotations declared there is nothing to sample and we return ``None`` so callers
        stay on the zero-overhead path.
        """
        params: Dict[str, EnvParam] = {}
        for name, spec in test.env_params.items():
            value = getattr(test.cmd_args, name, None)
            if not isinstance(value, list):
                continue
            encoding = spec.encoding
            if encoding is None:
                encoding = _infer_encoding(value)
            elif isinstance(encoding, LogEncoding):
                for c in value:
                    if not isinstance(c, (int, float)) or isinstance(c, bool):
                        raise TypeError(f"LogEncoding for '{name}' requires numeric candidates, got {type(c).__name__}")
                    if not math.isfinite(c) or c <= 0:
                        raise ValueError(f"LogEncoding for '{name}' requires strictly positive, finite candidates, got {c}")
            params[name] = EnvParam(candidates=value, weights=spec.weights, encoding=encoding)
        if not params:
            return None
        seed = int((test.agent_config or {}).get("random_seed", 0))
        return cls(params=params, seed=seed)

    def sample(self, trial: int) -> Dict[str, Any]:
        """
        Draw this trial's value for each parameter.

        Determinism: the same ``(seed, name, trial)`` yields the same draw across processes.
        Independence: each parameter's RNG is seeded ``f"{seed}:{name}:{trial}"`` so adding or
        removing one parameter never perturbs the others' draw sequences.
        """
        return {name: param.draw(random.Random(f"{self.seed}:{name}:{trial}")) for name, param in self.params.items()}

    def encode(self, regime: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a drawn regime (``{name: value}``) into one named observation leaf per parameter."""
        return {name: param.encode(regime[name]) for name, param in self.params.items()}

    def observation_descriptors(self) -> Dict[str, ObsLeafDescriptor]:
        """Per-parameter observation-leaf descriptors, keyed by parameter name."""
        return {name: param.observation_descriptor() for name, param in self.params.items()}


def write_env_params(path: Path, step: int, sample: Dict[str, Any]) -> None:
    """
    Append one trial's env_params sample to a step-aligned CSV.

    The CSV mirrors how ``trajectory.csv`` serialises its ``action`` column
    (one row per env.step(), sample dict stringified in a single cell) so the
    two files align 1:1 on ``step`` and a plain ``merge`` joins them.

    Empty samples are skipped, so a run without env_params writes nothing and
    callers can sink every trial unconditionally.
    """
    if step < 1:
        raise ValueError(f"step must be a positive trial index (cloudai DSE is 1-based); got {step}")
    if not sample:
        return
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(("step", "env"))
        writer.writerow([step, sample])


def validate_domain_randomization_active(test_scenario: "TestScenario") -> None:
    """
    Reject prepped configs that declare domain randomization no agent will run.

    env_params drive per-trial domain randomization, which only happens on a DSE run whose agent
    opts into sampling (``TestRun.is_domain_randomization_active``). Declaring env_params anywhere
    else is a silent no-op: domain randomization is enabled but never active. DSE-ness
    (``num_nodes``) and the agent both resolve only on the fully prepped config, so this is checked
    here rather than at parse time.
    """
    offenders = [
        tr.name
        for tr in test_scenario.test_runs
        if tr.test.is_domain_randomization_enabled and not tr.is_domain_randomization_active
    ]

    if offenders:
        raise TestScenarioParsingError(
            f"Tests {offenders} declare env_params but no agent will sample them. env_params are sampled "
            "per-trial only by a DSE run on an agent that opts into env_params sampling. Add a sweep "
            "(a list-valued cmd_args/extra_env_vars entry or num_nodes) and use such an agent, or remove "
            "env_params."
        )
