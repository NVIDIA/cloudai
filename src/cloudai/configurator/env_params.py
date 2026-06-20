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
Domain-randomization primitives for CloudAI DSE.

An env-randomized parameter is a workload knob whose value the environment
samples per trial (categorical, optional weights). It is sibling to
``cmd_args`` on a ``TestDefinition`` and does not enter the agent's action
space; the policy learns a robust mapping under that variation.

This module owns the data schema (``EnvParamSpec``), the deterministic
sampler (``EnvParamsSampler``), the persistence interface
(``EnvParamsSink`` + ``CsvSink``) and the per-step observer
(``EnvParamsObserver``). ``CloudAIGymEnv`` consumes these directly so the
artifacts (``env.csv``) and the cache key align 1:1 with ``trajectory.csv``
regardless of agent (PPO, BO, GA, MAB) or workload.
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
    """Specification of one env-randomized parameter (categorical)."""

    model_config = ConfigDict(extra="forbid")

    values: List[Any] = Field(
        min_length=2,
        description="Candidate values; a single-valued parameter is just a fixed cmd_args entry.",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional probability weights aligned with values; uniform if omitted.",
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        if self.weights is None:
            return self
        if len(self.weights) != len(self.values):
            raise ValueError(
                f"env_params weights length {len(self.weights)} does not match values length {len(self.values)}"
            )
        for w in self.weights:
            if not math.isfinite(w) or w < 0:
                raise ValueError(f"env_params weights must be finite and non-negative; got {w}")
        if sum(self.weights) <= 0:
            raise ValueError("env_params weights must have a positive sum")
        return self


class EnvParamsSampler:
    """
    Per-trial categorical sampler.

    Determinism contract: ``sample(t)`` returns the same dict on every call
    (across processes) for the same ``(seed, env_params, t)``.

    Independence contract: each parameter uses an RNG seeded by
    ``f"{seed}:{name}:{trial}"`` so adding or removing an unrelated
    parameter does not perturb existing parameters' draw sequences.
    """

    def __init__(self, env_params: Dict[str, EnvParamSpec], seed: int) -> None:
        self._env_params = env_params
        self._seed = seed

    def sample(self, trial: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, spec in self._env_params.items():
            rng = random.Random(f"{self._seed}:{name}:{trial}")
            if spec.weights is not None:
                out[name] = rng.choices(spec.values, weights=spec.weights, k=1)[0]
            else:
                out[name] = rng.choice(spec.values)
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

    Pre-step: samples ``test_run.test.env_params`` for ``test_run.step`` and
    stashes the result on ``test_run.current_env_params`` so the cache key and
    the workload's substitution both see it. Persistence is owned by
    ``CloudAIGymEnv.write_trajectory``, which sinks ``trajectory.csv`` and the
    ``env.csv`` projection from the single ``TrajectoryEntry`` - so a trial that
    writes no trajectory row writes no env.csv row either, keeping the two files
    1:1 step-aligned. Post-step: no-op.
    """

    def __init__(self, env_params: Dict[str, EnvParamSpec], seed: int) -> None:
        self._sampler = EnvParamsSampler(env_params, seed=seed)

    def before_step(self, test_run: Any) -> None:
        test_run.current_env_params = self._sampler.sample(test_run.step)

    def after_step(self, test_run: Any, observation: list, reward: float) -> None:
        del test_run, observation, reward  # persistence handled by CloudAIGymEnv.write_trajectory
