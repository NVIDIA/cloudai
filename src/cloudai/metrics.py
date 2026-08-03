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

"""Canonical metric observations and Speed-of-Light assessment models."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator


class OptimizationDirection(str, Enum):
    """Whether larger or smaller measured values are preferable."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TransferCoordinates(BaseModel):
    """Semantic coordinates for a point-to-point data transfer measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["default", "read", "write"] = "default"
    payload_size_bytes: int = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)


class CollectiveCoordinates(BaseModel):
    """Semantic coordinates for a collective communication measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collective: str = Field(min_length=1)
    placement: Literal["in_place", "out_of_place"]
    message_size_bytes: int = Field(gt=0)


class SOLTarget(BaseModel):
    """One SOL value and the coordinate selector under which it applies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("value")
    @classmethod
    def reject_non_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("SOL values must be finite")
        return value

    def selector(self) -> dict[str, Any]:
        """Return the explicitly configured coordinate subset for generic matching."""
        match = self.model_dump(mode="python", exclude={"value"}, exclude_none=True).get("match", {})
        if not isinstance(match, dict):
            raise TypeError(f"{type(self).__name__}.match must serialize to a dictionary")
        return match


class TransferMatch(BaseModel):
    """Optional transfer coordinates that select an SOL target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["default", "read", "write"] | None = None
    payload_size_bytes: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)


class TransferSOLTarget(SOLTarget):
    """SOL target selected by any explicitly configured transfer coordinates."""

    match: TransferMatch = Field(default_factory=TransferMatch)


class CollectiveMatch(BaseModel):
    """Optional collective coordinates that select an SOL target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collective: str | None = Field(default=None, min_length=1)
    placement: Literal["in_place", "out_of_place"] | None = None
    message_size_bytes: int | None = Field(default=None, gt=0)


class CollectiveSOLTarget(SOLTarget):
    """SOL target selected by any explicitly configured collective coordinates."""

    match: CollectiveMatch = Field(default_factory=CollectiveMatch)


@dataclass(frozen=True)
class MetricDefinition:
    """Static semantic definition of a metric produced by one or more workloads."""

    key: str
    display_name: str
    unit: str
    direction: OptimizationDirection
    coordinates_type: type[BaseModel]
    sol_target_type: type[SOLTarget]


TRANSFER_BANDWIDTH = MetricDefinition(
    key="transfer_bandwidth",
    display_name="Transfer bandwidth",
    unit="GB/s",
    direction=OptimizationDirection.MAXIMIZE,
    coordinates_type=TransferCoordinates,
    sol_target_type=TransferSOLTarget,
)
TRANSFER_LATENCY = MetricDefinition(
    key="transfer_latency",
    display_name="Transfer latency",
    unit="us",
    direction=OptimizationDirection.MINIMIZE,
    coordinates_type=TransferCoordinates,
    sol_target_type=TransferSOLTarget,
)
COLLECTIVE_BUS_BANDWIDTH = MetricDefinition(
    key="collective_bus_bandwidth",
    display_name="Collective bus bandwidth",
    unit="GB/s",
    direction=OptimizationDirection.MAXIMIZE,
    coordinates_type=CollectiveCoordinates,
    sol_target_type=CollectiveSOLTarget,
)
COLLECTIVE_LATENCY = MetricDefinition(
    key="collective_latency",
    display_name="Collective latency",
    unit="us",
    direction=OptimizationDirection.MINIMIZE,
    coordinates_type=CollectiveCoordinates,
    sol_target_type=CollectiveSOLTarget,
)


class MetricCatalog:
    """Registry of canonical metric definitions used to validate SOL configuration."""

    _metrics: ClassVar[dict[str, MetricDefinition]] = {
        metric.key: metric
        for metric in (
            TRANSFER_BANDWIDTH,
            TRANSFER_LATENCY,
            COLLECTIVE_BUS_BANDWIDTH,
            COLLECTIVE_LATENCY,
        )
    }

    @classmethod
    def get(cls, key: str) -> MetricDefinition:
        try:
            return cls._metrics[key]
        except KeyError as exc:
            available = ", ".join(sorted(cls._metrics))
            raise ValueError(f"Unknown SOL metric '{key}'. Available metrics: {available}") from exc


MetricSOLConfig = dict[str, list[SerializeAsAny[SOLTarget]]]


def _selectors_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether two selectors can match the same coordinates."""
    return all(first[key] == second[key] for key in first.keys() & second.keys())


def _validate_unambiguous_targets(metric_key: str, targets: list[SOLTarget]) -> None:
    """Reject targets for which specificity cannot determine a unique winner."""
    selectors = [target.selector() for target in targets]
    for index, selector in enumerate(selectors):
        for other in selectors[index + 1 :]:
            if len(selector) == len(other) and _selectors_overlap(selector, other):
                raise ValueError(
                    f"Ambiguous SOL targets for metric '{metric_key}': {selector} and {other} "
                    "can match the same observation with equal specificity"
                )


def parse_sol_spec(value: dict[str, Any] | None) -> MetricSOLConfig:
    """Parse metric-keyed SOL target lists using each metric's target schema."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Structured SOL configuration must be a dictionary")
    parsed: MetricSOLConfig = {}
    for key, raw_targets in value.items():
        metric = MetricCatalog.get(key)
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"SOL metric '{key}' must contain a non-empty list of targets")
        targets = [metric.sol_target_type.model_validate(target) for target in raw_targets]
        _validate_unambiguous_targets(key, targets)
        parsed[key] = targets
    return parsed


def merge_sol_configs(*configs: MetricSOLConfig | None) -> MetricSOLConfig:
    """Merge SOL configurations by metric key; later levels replace a complete metric."""
    merged: MetricSOLConfig = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def resolve_sol(targets: list[SOLTarget], coordinates: BaseModel) -> float | None:
    """Resolve the most specific SOL target matching the observation coordinates."""
    coordinate_values = coordinates.model_dump(mode="python")
    matches = [
        target
        for target in targets
        if all(coordinate_values.get(key) == value for key, value in target.selector().items())
    ]
    if not matches:
        return None
    return max(matches, key=lambda target: len(target.selector())).value


@dataclass(frozen=True)
class MetricObservation:
    """One finite measured value at typed semantic coordinates."""

    metric: MetricDefinition
    value: float
    coordinates: BaseModel

    def __post_init__(self) -> None:
        """Validate that the value and coordinates match the metric definition."""
        if not math.isfinite(self.value):
            raise ValueError(f"Metric '{self.metric.key}' observation must be finite")
        if not isinstance(self.coordinates, self.metric.coordinates_type):
            raise TypeError(
                f"Metric '{self.metric.key}' requires {self.metric.coordinates_type.__name__}, "
                f"got {type(self.coordinates).__name__}"
            )


@dataclass(frozen=True)
class MetricAssessment:
    """A measured observation enriched with its resolved SOL and normalized attainment."""

    observation: MetricObservation
    sol: float | None
    attainment: float | None
    gap: float | None


@dataclass(frozen=True)
class MetricAssessmentSummary:
    """Compact distribution summary for one metric in one test run."""

    metric: MetricDefinition
    observations: int
    matched: int
    worst_attainment: float | None
    median_attainment: float | None
    best_attainment: float | None


def assess_observation(observation: MetricObservation, sol_config: MetricSOLConfig) -> MetricAssessment:
    """Resolve and compare one observation against the configured SOL for its metric."""
    targets = sol_config.get(observation.metric.key)
    sol = resolve_sol(targets, observation.coordinates) if targets else None
    if sol is None:
        return MetricAssessment(observation=observation, sol=None, attainment=None, gap=None)

    attainment = (
        observation.value / sol
        if observation.metric.direction is OptimizationDirection.MAXIMIZE
        else sol / observation.value
    )
    return MetricAssessment(
        observation=observation,
        sol=sol,
        attainment=attainment,
        gap=observation.value - sol,
    )


def assess_test_run_metrics(system: Any, test_run: Any) -> list[MetricAssessment]:
    """Collect and assess every canonical metric produced by a completed test run."""
    observations = test_run.test.metric_observations(system, test_run)
    return [assess_observation(observation, test_run.metric_sol) for observation in observations]


def summarize_assessments(assessments: list[MetricAssessment]) -> list[MetricAssessmentSummary]:
    """Summarize SOL coverage and attainment by canonical metric."""
    grouped: dict[str, list[MetricAssessment]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment.observation.metric.key, []).append(assessment)

    summaries: list[MetricAssessmentSummary] = []
    for metric_assessments in grouped.values():
        attainments = [item.attainment for item in metric_assessments if item.attainment is not None]
        summaries.append(
            MetricAssessmentSummary(
                metric=metric_assessments[0].observation.metric,
                observations=len(metric_assessments),
                matched=len(attainments),
                worst_attainment=min(attainments) if attainments else None,
                median_attainment=statistics.median(attainments) if attainments else None,
                best_attainment=max(attainments) if attainments else None,
            )
        )
    return summaries


def coordinates_dict(coordinates: BaseModel) -> dict[str, Any]:
    """Return coordinates as a serialization-safe dictionary."""
    return coordinates.model_dump(mode="json")
