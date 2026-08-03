# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metric observations and Speed-of-Light assessment models."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


class OptimizationDirection(StrEnum):
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


class TransferSOL(BaseModel):
    """Scalar SOL targets for transfer directions, in the metric's canonical unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    read: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    write: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("default", "read", "write")
    @classmethod
    def reject_non_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("SOL values must be finite")
        return value


class CollectivePlacementSOL(BaseModel):
    """Scalar SOL targets for the two NCCL placement modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    in_place: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    out_of_place: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class CollectiveSOL(RootModel[dict[str, CollectivePlacementSOL]]):
    """SOL targets keyed by normalized collective name, with optional ``default``."""

    @field_validator("root")
    @classmethod
    def reject_blank_collective_names(
        cls, value: dict[str, CollectivePlacementSOL]
    ) -> dict[str, CollectivePlacementSOL]:
        if any(not name.strip() for name in value):
            raise ValueError("Collective SOL names must be non-blank")
        return value


SOLResolver = Callable[[BaseModel, BaseModel], float | None]


@dataclass(frozen=True)
class MetricDefinition:
    """Static semantic definition of a metric produced by one or more workloads."""

    key: str
    display_name: str
    unit: str
    direction: OptimizationDirection
    coordinates_type: type[BaseModel]
    sol_type: type[BaseModel]
    resolve_sol: SOLResolver


def _resolve_transfer_sol(config: BaseModel, coordinates: BaseModel) -> float | None:
    assert isinstance(config, TransferSOL)
    assert isinstance(coordinates, TransferCoordinates)
    return getattr(config, coordinates.operation) or config.default


def _resolve_collective_sol(config: BaseModel, coordinates: BaseModel) -> float | None:
    assert isinstance(config, CollectiveSOL)
    assert isinstance(coordinates, CollectiveCoordinates)
    targets = config.root.get(coordinates.collective) or config.root.get("default")
    return getattr(targets, coordinates.placement) if targets else None


TRANSFER_BANDWIDTH = MetricDefinition(
    key="transfer_bandwidth",
    display_name="Transfer bandwidth",
    unit="GB/s",
    direction=OptimizationDirection.MAXIMIZE,
    coordinates_type=TransferCoordinates,
    sol_type=TransferSOL,
    resolve_sol=_resolve_transfer_sol,
)
TRANSFER_LATENCY = MetricDefinition(
    key="transfer_latency",
    display_name="Transfer latency",
    unit="us",
    direction=OptimizationDirection.MINIMIZE,
    coordinates_type=TransferCoordinates,
    sol_type=TransferSOL,
    resolve_sol=_resolve_transfer_sol,
)
COLLECTIVE_BUS_BANDWIDTH = MetricDefinition(
    key="collective_bus_bandwidth",
    display_name="Collective bus bandwidth",
    unit="GB/s",
    direction=OptimizationDirection.MAXIMIZE,
    coordinates_type=CollectiveCoordinates,
    sol_type=CollectiveSOL,
    resolve_sol=_resolve_collective_sol,
)
COLLECTIVE_LATENCY = MetricDefinition(
    key="collective_latency",
    display_name="Collective latency",
    unit="us",
    direction=OptimizationDirection.MINIMIZE,
    coordinates_type=CollectiveCoordinates,
    sol_type=CollectiveSOL,
    resolve_sol=_resolve_collective_sol,
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


MetricSOLConfig = dict[str, BaseModel]


def parse_sol_spec(value: dict[str, Any] | None) -> MetricSOLConfig:
    """Parse a metric-keyed SOL dictionary using each metric's own schema."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Structured SOL configuration must be a dictionary")
    return {key: MetricCatalog.get(key).sol_type.model_validate(config) for key, config in value.items()}


def merge_sol_configs(*configs: MetricSOLConfig | None) -> MetricSOLConfig:
    """Merge SOL configurations by metric key; later levels replace a complete metric."""
    merged: MetricSOLConfig = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


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
    metric_config = sol_config.get(observation.metric.key)
    sol = observation.metric.resolve_sol(metric_config, observation.coordinates) if metric_config else None
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
