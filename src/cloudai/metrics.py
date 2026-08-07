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

"""Semantic metric observations and Speed-of-Light assessment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

MetricValue: TypeAlias = str | int | float | bool
MetricDimensions: TypeAlias = Mapping[str, MetricValue]


class OptimizationDirection(str, Enum):
    """Whether larger or smaller measured values are preferable."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class DimensionDefinition:
    """A typed semantic dimension shared by metric observations and SOL selectors."""

    key: str
    label: str
    value_type: Any

    def validate(self, value: Any) -> MetricValue:
        """Validate one configured or observed dimension value."""
        return TypeAdapter(self.value_type).validate_python(value)


@dataclass(frozen=True)
class MetricDefinition:
    """A measured quantity independent of the workload that produced it."""

    key: str
    display_name: str
    unit: str
    direction: OptimizationDirection


class MetricCatalog:
    """Registry of metrics and dimensions supported by structured SOL configuration."""

    _metrics: ClassVar[dict[str, MetricDefinition]] = {}
    _dimensions: ClassVar[dict[str, DimensionDefinition]] = {}

    @classmethod
    def metric(cls, key: str) -> MetricDefinition:
        try:
            return cls._metrics[key]
        except KeyError as exc:
            available = ", ".join(sorted(cls._metrics))
            raise ValueError(f"Unknown SOL metric '{key}'. Available metrics: {available}") from exc

    @classmethod
    def dimension(cls, key: str) -> DimensionDefinition:
        try:
            return cls._dimensions[key]
        except KeyError as exc:
            available = ", ".join(sorted(cls._dimensions))
            raise ValueError(f"Unknown metric dimension '{key}'. Available dimensions: {available}") from exc

    @classmethod
    def validate_dimensions(cls, values: Mapping[str, Any]) -> MetricDimensions:
        return {key: cls.dimension(key).validate(value) for key, value in values.items()}


SIZE_BYTES = DimensionDefinition("size_bytes", "Size", Annotated[int, Field(strict=True, ge=0)])
BATCH_SIZE = DimensionDefinition("batch_size", "Batch size", Annotated[int, Field(strict=True, gt=0)])
OPERATION = DimensionDefinition("operation", "Operation", Annotated[str, Field(strict=True, min_length=1)])
PLACEMENT = DimensionDefinition("placement", "Placement", Literal["in_place", "out_of_place"])
BANDWIDTH_BASIS = DimensionDefinition("bandwidth_basis", "Bandwidth basis", Literal["bus", "payload"])
BACKEND = DimensionDefinition("backend", "Backend", Annotated[str, Field(strict=True, min_length=1)])
SOURCE_MEMORY = DimensionDefinition("source_memory", "Source memory", Annotated[str, Field(strict=True, min_length=1)])
TARGET_MEMORY = DimensionDefinition("target_memory", "Target memory", Annotated[str, Field(strict=True, min_length=1)])

BANDWIDTH = MetricDefinition(
    key="bandwidth",
    display_name="Bandwidth",
    unit="GB/s",
    direction=OptimizationDirection.MAXIMIZE,
)
LATENCY = MetricDefinition(
    key="latency",
    display_name="Latency",
    unit="us",
    direction=OptimizationDirection.MINIMIZE,
)

MetricCatalog._metrics = {metric.key: metric for metric in (BANDWIDTH, LATENCY)}
MetricCatalog._dimensions = {
    dimension.key: dimension
    for dimension in (
        SIZE_BYTES,
        BATCH_SIZE,
        OPERATION,
        PLACEMENT,
        BANDWIDTH_BASIS,
        BACKEND,
        SOURCE_MEMORY,
        TARGET_MEMORY,
    )
}


class SOLTarget(BaseModel):
    """One SOL value and the observation dimensions under which it applies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(gt=0, allow_inf_nan=False)
    match: MetricDimensions = Field(default_factory=dict)

    @field_validator("match", mode="before")
    @classmethod
    def validate_match(cls, value: dict[str, Any] | None) -> MetricDimensions:
        return MetricCatalog.validate_dimensions(value or {})


MetricSOLConfig: TypeAlias = dict[str, list[SOLTarget]]


def _selectors_overlap(first: MetricDimensions, second: MetricDimensions) -> bool:
    return all(first[key] == second[key] for key in first.keys() & second.keys())


def parse_sol_spec(value: dict[str, Any] | None) -> MetricSOLConfig:
    """Parse and validate metric-keyed SOL targets."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Structured SOL configuration must be a dictionary")

    parsed: MetricSOLConfig = {}
    for metric_key, raw_targets in value.items():
        MetricCatalog.metric(metric_key)
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"SOL metric '{metric_key}' must contain a non-empty list of targets")
        targets = [SOLTarget.model_validate(target) for target in raw_targets]
        for index, target in enumerate(targets):
            for other in targets[index + 1 :]:
                if len(target.match) == len(other.match) and _selectors_overlap(target.match, other.match):
                    raise ValueError(
                        f"Ambiguous SOL targets for metric '{metric_key}': {target.match} and {other.match} "
                        "can match the same observation with equal specificity"
                    )
        parsed[metric_key] = targets
    return parsed


def merge_sol_configs(*configs: MetricSOLConfig | None) -> MetricSOLConfig:
    """Merge SOL configurations; a more specific level replaces an entire metric."""
    merged: MetricSOLConfig = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


@dataclass(frozen=True)
class MetricObservation:
    """One finite measured value and its dimensions."""

    metric: MetricDefinition
    value: float
    dimensions: MetricDimensions

    def __post_init__(self) -> None:
        """Validate the measurement and normalize its dimensions."""
        if not math.isfinite(self.value):
            raise ValueError(f"Metric '{self.metric.key}' observation must be finite")
        dimensions = MetricCatalog.validate_dimensions(self.dimensions)
        object.__setattr__(self, "dimensions", dimensions)


@dataclass(frozen=True)
class MetricAssessment:
    """A metric observation with its applicable SOL comparison."""

    observation: MetricObservation
    target: SOLTarget | None

    @property
    def sol(self) -> float | None:
        """Return the applicable SOL value."""
        return self.target.value if self.target is not None else None

    @property
    def attainment(self) -> float | None:
        """Return the measured performance relative to SOL."""
        if self.target is None:
            return None
        if self.observation.metric.direction is OptimizationDirection.MAXIMIZE:
            return self.observation.value / self.target.value
        if self.observation.value == 0:
            return None
        return self.target.value / self.observation.value


def assess_observation(observation: MetricObservation, sol_config: MetricSOLConfig) -> MetricAssessment:
    """Resolve the most specific target and assess one observation."""
    targets = sol_config.get(observation.metric.key, [])
    matches = [
        target
        for target in targets
        if all(observation.dimensions.get(key) == value for key, value in target.match.items())
    ]
    target = max(matches, key=lambda item: len(item.match)) if matches else None
    return MetricAssessment(observation, target)


def assess_test_run_metrics(system: Any, test_run: Any) -> list[MetricAssessment]:
    """Collect and assess every metric produced by a completed test run."""
    observations = test_run.test.metric_observations(system, test_run)
    return [assess_observation(observation, test_run.metric_sol) for observation in observations]


def dimension_label(key: str) -> str:
    """Return a human-readable label for a registered dimension."""
    return MetricCatalog.dimension(key).label


def format_dimension(key: str, value: object) -> str:
    """Format one semantic dimension value for reports."""
    if key == SIZE_BYTES.key:
        try:
            size = float(int(str(value)))
        except ValueError:
            return str(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:g}{unit}"
            size /= 1024
    return str(value).replace("_", " ")
