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
such a field: ``env_params.<name>`` reclassifies ``cmd_args.<name>`` from
action-space to env-sampled and carries only *how* to sample (optional
``weights``), never the values. The policy learns a robust mapping under that
variation; the knob never enters the agent's action space.

This module owns the annotation schema (``EnvParamSpec``), the deterministic
sampler (``EnvParamsSampler``), the persistence interface
(``EnvParamsSink`` + ``CsvSink``) and the per-step observer
(``EnvParamsObserver``). ``CloudAIGymEnv`` consumes these directly so the
artifacts (``env.csv``) and the cache key align 1:1 with ``trajectory.csv``
regardless of agent or workload.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


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


class EnvParamsSampler:
    """
    Per-trial categorical sampler over candidate lists.

    Candidates are resolved from ``cmd_args`` by the caller and passed in as
    ``{name: [v0, v1, ...]}``; optional ``weights`` mirror that mapping.

    Determinism contract: ``sample(t)`` returns the same dict on every call
    (across processes) for the same ``(seed, candidates, t)``.

    Independence contract: each parameter uses an RNG seeded by
    ``f"{seed}:{name}:{trial}"`` so adding or removing an unrelated
    parameter does not perturb existing parameters' draw sequences.
    """

    def __init__(
        self,
        candidates: Dict[str, List[Any]],
        seed: int,
        weights: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self._candidates = candidates
        self._weights = weights or {}
        self._seed = seed

    def sample(self, trial: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, choices in self._candidates.items():
            rng = random.Random(f"{self._seed}:{name}:{trial}")
            weights = self._weights.get(name)
            if weights is not None:
                out[name] = rng.choices(choices, weights=weights, k=1)[0]
            else:
                out[name] = rng.choice(choices)
        return out


@runtime_checkable
class EnvParamsSink(Protocol):
    """Persist one trial's env_params sample; empty samples must be no-ops."""

    def write(self, step: int, sample: Dict[str, Any]) -> None: ...


class CsvSink:
    """
    Append per-trial env_params samples to a step-aligned CSV.

    The CSV mirrors how ``trajectory.csv`` serialises its ``action`` column
    (one row per env.step(), sample dict stringified in a single cell) so the
    two files align 1:1 on ``step`` and a plain ``merge`` joins them.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, step: int, sample: Dict[str, Any]) -> None:
        if step < 1:
            raise ValueError(f"step must be a positive trial index (cloudai DSE is 1-based); got {step}")
        if not sample:
            return
        new_file = not self._path.exists()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(("step", "env"))
            writer.writerow([step, sample])


@runtime_checkable
class StepObserver(Protocol):
    """
    Hook fired by ``CloudAIGymEnv.step()`` around each trial.

    ``before_step`` runs before the cache lookup and before any workload
    execution. ``after_step`` runs after the trajectory row is written.
    """

    def before_step(self, test_run: Any) -> None: ...

    def after_step(self, test_run: Any, observation: list, reward: float) -> None: ...


class EnvParamsObserver:
    """
    Sample env_params per step and stash them for the cache and the workload.

    Construction: resolves each annotated knob's candidate list from
    ``cmd_args`` (the single source of truth) once, up front. A knob whose
    ``cmd_args`` value is a scalar (not a list) is fixed - its annotation is a
    no-op and it is skipped, so nothing is sampled or stashed for it.

    Pre-step: samples the resolved candidates for ``test_run.step`` and stashes
    the result on ``test_run.current_env_params`` so the cache key and the
    cmd_args overlay both see it. Persistence is owned by
    ``CloudAIGymEnv.write_trajectory``, which sinks ``trajectory.csv`` and the
    ``env.csv`` projection from the single ``TrajectoryEntry`` - so a trial that
    writes no trajectory row writes no env.csv row either, keeping the two files
    1:1 step-aligned. Post-step: no-op.
    """

    def __init__(self, env_params: Dict[str, EnvParamSpec], cmd_args: Any, seed: int) -> None:
        candidates: Dict[str, List[Any]] = {}
        weights: Dict[str, List[float]] = {}
        for name, spec in env_params.items():
            value = getattr(cmd_args, name, None)
            if not isinstance(value, list):
                continue  # scalar cmd_args knob is fixed; the annotation is a no-op
            candidates[name] = value
            if spec.weights is not None:
                weights[name] = spec.weights
        self._sampler = EnvParamsSampler(candidates, seed=seed, weights=weights)

    def before_step(self, test_run: Any) -> None:
        test_run.current_env_params = self._sampler.sample(test_run.step)

    def after_step(self, test_run: Any, observation: list, reward: float) -> None:
        del test_run, observation, reward  # persistence handled by CloudAIGymEnv.write_trajectory
