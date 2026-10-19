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

"""Scenario-level Pareto plots from semantic metric observations."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import toml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

import cloudai.metrics as metrics
from cloudai.core import Reporter, TestRun
from cloudai.models.scenario import ReportConfig, TestRunDetails


class MetricSelector(BaseModel):
    """Select exactly one observation per execution."""

    model_config = ConfigDict(extra="forbid")
    metric: str
    match: dict[str, metrics.MetricValue] = Field(default_factory=dict)
    direction: metrics.OptimizationDirection | None = None

    @field_validator("metric")
    @classmethod
    def known_metric(cls, value: str) -> str:
        metrics.MetricCatalog.get_metric(value)
        return value

    @field_validator("match")
    @classmethod
    def known_dimensions(cls, value: dict[str, metrics.MetricValue]) -> dict[str, metrics.MetricValue]:
        return dict(metrics.MetricCatalog.validate_dimensions(value))

    @property
    def definition(self) -> metrics.MetricDefinition:
        return metrics.MetricCatalog.get_metric(self.metric)

    @property
    def resolved_direction(self) -> metrics.OptimizationDirection:
        return self.direction or self.definition.direction

    @property
    def label(self) -> str:
        qualifiers = ", ".join(f"{key}={value}" for key, value in self.match.items())
        name = self.definition.display_name + (f" · {qualifiers}" if qualifiers else "")
        return f"{name} ({self.definition.unit}) · {self.resolved_direction.value}"

    def select(self, observations: list[metrics.MetricObservation]) -> metrics.MetricObservation:
        selected = [
            obs
            for obs in observations
            if obs.metric.key == self.metric and all(obs.dimensions.get(k) == v for k, v in self.match.items())
        ]
        if not selected:
            raise ValueError(f"Missing metric '{self.metric}' matching {self.match}")
        if len(selected) != 1:
            raise ValueError(
                f"Ambiguous metric '{self.metric}': {len(selected)} observations; specify match dimensions"
            )
        obs = selected[0]
        if not math.isfinite(obs.value) or obs.metric.unit != self.definition.unit:
            raise ValueError(f"Invalid value or inconsistent unit for '{self.metric}'")
        return obs


class ParetoFrontierConfig(BaseModel):
    """Case IDs whose executions compete within one frontier."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    cases: list[str] = Field(min_length=1)
    match: dict[str, metrics.MetricValue] = Field(default_factory=dict)

    @field_validator("match")
    @classmethod
    def known_dimensions(cls, value: dict[str, metrics.MetricValue]) -> dict[str, metrics.MetricValue]:
        return dict(metrics.MetricCatalog.validate_dimensions(value))


class ParetoPlotConfig(BaseModel):
    """Two objectives and the independent frontiers displayed together."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    x: MetricSelector
    y: MetricSelector
    frontiers: list[ParetoFrontierConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_frontiers(self) -> Self:
        names = [frontier.name for frontier in self.frontiers]
        if len(set(names)) != len(names):
            raise ValueError("Frontier names must be unique within each plot")
        return self


class ParetoReportConfig(ReportConfig):
    """Opt-in plots configured at scenario level."""

    plots: list[ParetoPlotConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def enabled_plots(self) -> Self:
        if self.enable and not self.plots:
            raise ValueError("Pareto reporting requires at least one plot")
        return self


def pareto_mask(
    points: Sequence[tuple[float, float]],
    x_direction: metrics.OptimizationDirection,
    y_direction: metrics.OptimizationDirection,
) -> list[bool]:
    """
    Identify non-dominated points, retaining ties and original order.

    Sort by descending desirability on both axes, then scan the best y seen.
    Equal coordinates share dominance status, preserving their provenance.
    """
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("Pareto coordinates must be finite")
    sx = 1 if x_direction == metrics.OptimizationDirection.MAXIMIZE else -1
    sy = 1 if y_direction == metrics.OptimizationDirection.MAXIMIZE else -1
    ordered = sorted({(sx * x, sy * y) for x, y in points}, reverse=True)
    frontier: set[tuple[float, float]] = set()
    best_y = -math.inf
    for x, y in ordered:
        if y > best_y:
            frontier.add((x, y))
            best_y = y
    return [(sx * x, sy * y) in frontier for x, y in points]


@dataclass
class RunObservations:
    """Measurements and executed configuration, or a collection failure."""

    tr: TestRun
    observations: list[metrics.MetricObservation] = field(default_factory=list)
    error: str = ""
    config: str = ""

    @property
    def label(self) -> str:
        return f"{self.tr.name} · iteration={self.tr.current_iteration} · step={self.tr.step}"


def numeric_directories(root: Path) -> list[Path]:
    return sorted((p for p in root.glob("*") if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))


class ParetoReporter(Reporter):
    """Render all successful iterations and search steps in selected cases."""

    config: ParetoReportConfig

    def collect_runs(self) -> list[RunObservations]:
        selected_cases = {case for plot in self.config.plots for frontier in plot.frontiers for case in frontier.cases}
        known_cases = {tr.name for tr in self.test_scenario.test_runs}
        if unknown := selected_cases - known_cases:
            raise ValueError(f"Unknown Pareto case IDs: {', '.join(sorted(unknown))}")
        runs = []
        for case in self.test_scenario.test_runs:
            if case.name not in selected_cases:
                continue
            iterations = numeric_directories(self.results_root / case.name)
            if not iterations:
                runs.append(RunObservations(case, error="No execution results found"))
            for iteration in iterations:
                paths = numeric_directories(iteration) if case.is_dse_job else [iteration]
                if not paths:
                    runs.append(RunObservations(case, error=f"No search steps found in iteration {iteration.name}"))
                for path in paths:
                    tr = copy.deepcopy(case)
                    tr.current_iteration = int(iteration.name)
                    tr.step = int(path.name) if case.is_dse_job else 0
                    tr.output_path = path
                    runs.append(self._collect_run(tr, case.is_dse_job))
        return runs

    def _collect_run(self, tr: TestRun, is_search: bool) -> RunObservations:
        run = RunObservations(tr)
        try:
            dump = tr.output_path / "test-run.toml"
            if dump.exists():
                details = TestRunDetails.model_validate(toml.load(dump))
                if details.test_definition is None:
                    raise ValueError("Missing executed test definition")
                tr.test = details.test_definition
                tr.num_nodes = details.nnodes
                tr.nodes = details.nodes
            elif is_search:
                raise ValueError("Missing test-run.toml for search step")
            run.config = toml.dumps(
                {
                    "num_nodes": tr.nnodes,
                    "test": tr.test.model_dump(mode="json", by_alias=True, exclude_none=True),
                }
            )
            status = tr.test.was_run_successful(tr)
            if not status.is_successful:
                raise ValueError(status.error_message or "Execution failed or incomplete")
            run.observations = tr.test.metric_observations(self.system, tr)
        except Exception as exc:
            run.error = str(exc)
            logging.debug("Cannot collect Pareto results at %s: %s", tr.output_path, exc)
        return run

    @staticmethod
    def select_pair(
        run: RunObservations, plot: ParetoPlotConfig, frontier: ParetoFrontierConfig
    ) -> tuple[metrics.MetricObservation, metrics.MetricObservation]:
        if run.error:
            raise ValueError(run.error)
        observations = [
            obs for obs in run.observations if all(obs.dimensions.get(k) == v for k, v in frontier.match.items())
        ]
        x, y = plot.x.select(observations), plot.y.select(observations)
        # Statistics can intentionally differ, e.g. mean TTFT versus p99 TPOT.
        shared = (x.dimensions.keys() & y.dimensions.keys()) - {"statistic"}
        if any(x.dimensions[key] != y.dimensions[key] for key in shared):
            raise ValueError("Axis metrics have incompatible measurement dimensions")
        return x, y

    def build_sections(self, runs: list[RunObservations]) -> list[dict[str, Any]]:
        sections = []
        for plot_index, plot in enumerate(self.config.plots):
            datasets: list[dict[str, Any]] = []
            points: list[dict[str, Any]] = []
            excluded: list[dict[str, str]] = []
            summaries = []
            for group_index, frontier in enumerate(plot.frontiers):
                candidates = []
                excluded_before = len(excluded)
                for run in runs:
                    if run.tr.name not in frontier.cases:
                        continue
                    try:
                        x, y = self.select_pair(run, plot, frontier)
                    except ValueError as exc:
                        excluded.append({"group": frontier.name, "run": run.label, "reason": str(exc)})
                        continue
                    dimensions = ", ".join(
                        f"{axis}.{key}={value}"
                        for axis, observation in (("x", x), ("y", y))
                        for key, value in observation.dimensions.items()
                    )
                    candidates.append(
                        {
                            "x": x.value,
                            "y": y.value,
                            "run": run.label,
                            "group": frontier.name,
                            "dimensions": dimensions,
                            "config": run.config,
                            "artifact": run.tr.output_path.relative_to(self.results_root).as_posix(),
                        }
                    )
                mask = pareto_mask(
                    [(point["x"], point["y"]) for point in candidates],
                    plot.x.resolved_direction,
                    plot.y.resolved_direction,
                )
                for point, is_frontier in zip(candidates, mask, strict=True):
                    point["status"] = "Frontier" if is_frontier else "Dominated"
                for status in ("Frontier", "Dominated"):
                    data = [
                        {k: v for k, v in point.items() if k != "config"}
                        for point in candidates
                        if point["status"] == status
                    ]
                    if data:
                        datasets.append(
                            {
                                "label": f"{frontier.name} · {status.lower()}",
                                "data": sorted(data, key=lambda p: (p["x"], p["y"])),
                                "source_color_index": group_index,
                                "is_frontier": status == "Frontier",
                            }
                        )
                summaries.append(
                    {
                        "name": frontier.name,
                        "candidates": len(candidates),
                        "frontier": sum(mask),
                        "excluded": len(excluded) - excluded_before,
                    }
                )
                points.extend(candidates)
            sections.append(
                {
                    "pareto": True,
                    "title": plot.name,
                    "group_name": "Each frontier compares only its selected cases. Lines connect measured points.",
                    "summaries": summaries,
                    "points": points,
                    "excluded": excluded,
                    "chart": {
                        "id": f"pareto-{plot_index}",
                        "type": "scatter",
                        "pareto": True,
                        "x_axis_type": "linear",
                        "y_axis_type": "linear",
                        "x_axis_label": plot.x.label,
                        "y_axis_label": plot.y.label,
                        "datasets": datasets,
                    },
                }
            )
        return sections

    def generate(self) -> None:
        sections = self.build_sections(self.collect_runs())
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent.parent / "util"),
            autoescape=jinja2.select_autoescape(enabled_extensions=("html", "htm", "xml", "jinja2")),
        )
        content = env.get_template("comparison-report-v2.jinja2").render(
            name=f"{self.test_scenario.name} Pareto Report",
            sections=sections,
        )
        path = self.results_root / "pareto.html"
        self.results_root.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logging.info("Pareto report created: %s", path)
