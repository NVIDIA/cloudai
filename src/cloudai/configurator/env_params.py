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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from cloudai._core.exceptions import TestScenarioParsingError
from cloudai._core.registry import Registry

if TYPE_CHECKING:
    from cloudai._core.test_scenario import TestScenario


class EnvParamSpec(BaseModel):
    """
    Annotation marking one cmd_args field as env-sampled.

    Carries only *how* to sample - the candidate values themselves live in
    ``cmd_args.<name>`` as a plain list. ``weights`` (optional) are positional,
    aligned 1:1 with that candidate list; omit for uniform sampling. The
    length match against the candidate list is a cross-field check enforced by
    ``TestDefinition`` (which can see ``cmd_args``); here we validate only the
    weights' intrinsic shape.
    """

    model_config = ConfigDict(extra="forbid")

    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional probability weights aligned with the cmd_args candidate list; uniform if omitted.",
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


class ObsLeafDescriptor(BaseModel):
    """
    Description of one leaf of a structured (named) observation.

    A structured observation maps each observed name to a self-describing leaf
    so adapters can build the matching subspace without guessing: a ``"box"``
    leaf becomes a continuous vector of width ``dim`` (e.g. a log-encoded
    env_param as ``dim=2``); a ``"discrete"`` leaf becomes a categorical of
    size ``n``. Stateless agents that consume the flat observation ignore this.
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


@runtime_checkable
class StructuredObservation(Protocol):
    """
    Optional env hooks that expose a structured (per-leaf) observation.

    An env opts in by returning per-leaf :class:`ObsLeafDescriptor` from
    ``structured_observation_descriptors`` (``None`` keeps the flat-vector
    path) and encoding a raw observation into the matching named leaves via
    ``encode_observation``. ``GymnasiumAdapter`` consumes these to expose a
    ``gymnasium.spaces.Dict`` observation; the hooks are duck-typed, so envs
    need not subclass this Protocol.
    """

    def structured_observation_descriptors(self) -> Optional[Dict[str, ObsLeafDescriptor]]: ...

    def encode_observation(self, observation: list) -> Dict[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class EnvParam:
    """
    One env-sampled knob, resolved from cmd_args: its candidate values and optional weights.

    Weights (when present) are positional, aligned 1:1 with ``candidates``; ``None`` means
    uniform sampling. Keeping the two together makes each knob a self-contained draw.
    """

    candidates: List[Any]
    weights: Optional[List[float]] = None

    def draw(self, rng: random.Random) -> Any:
        if self.weights is not None:
            return rng.choices(self.candidates, weights=self.weights, k=1)[0]
        return rng.choice(self.candidates)


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
    def from_test(cls, test: Any) -> Optional["EnvParams"]:
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
            params[name] = EnvParam(candidates=value, weights=spec.weights)
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


class EnvParamsSink:
    """
    Append per-trial env_params samples to a step-aligned CSV.

    The CSV mirrors how ``trajectory.csv`` serialises its ``action`` column
    (one row per env.step(), sample dict stringified in a single cell) so the
    two files align 1:1 on ``step`` and a plain ``merge`` joins them.

    Empty samples are skipped, so a run without env_params writes nothing and
    callers can sink every trial unconditionally.
    """

    def write(self, path: Path, step: int, sample: Dict[str, Any]) -> None:
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


def validate_dse_env_params(test_scenario: "TestScenario") -> None:
    """
    Reject prepped configs that declare env_params nothing will sample.

    env_params are sampled per-trial by CloudAIGymEnv on an agent-driven run: a DSE sweep on an agent
    that opts in via ``BaseAgent.samples_env_params``, or an online live-RL run (``cmd_args.live_rl_mode``),
    which drives the agent's own loop and samples regardless of agent kind. A plain run has no per-trial
    loop, and a non-opting agent ignores env_params, so declaring them there is a silent no-op. is_dse_job,
    the agent, and live_rl_mode all resolve only on the fully prepped config, so this is validated here
    rather than at parse time.
    """
    agents = Registry().agents_map

    offenders = []
    for tr in test_scenario.test_runs:
        if not tr.test.env_params:
            continue

        agent = agents.get(tr.test.agent)
        live_rl = bool(getattr(tr.test.cmd_args, "live_rl_mode", False))
        # Unknown agent: defer to the dedicated agent-resolution error rather than masking it here.
        sampled = live_rl or (tr.is_dse_job and (agent is None or agent.samples_env_params))
        if not sampled:
            offenders.append(tr.name)

    if offenders:
        raise TestScenarioParsingError(
            f"Tests {offenders} declare env_params but no agent will sample them. env_params are sampled "
            "per-trial only on an agent-driven run: a DSE sweep on an agent that opts into env_params "
            "sampling, or cmd_args.live_rl_mode. Add a sweep (a list-valued cmd_args/extra_env_vars entry "
            "or num_nodes) with such an agent, set live_rl_mode, or remove env_params."
        )
