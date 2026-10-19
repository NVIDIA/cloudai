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

import itertools
import random

import pytest
from pydantic import ValidationError

import cloudai.metrics as metrics
from cloudai.core import TestRun
from cloudai.models.scenario import parse_reports_spec
from cloudai.report_generator.pareto_report import (
    MetricSelector,
    ParetoPlotConfig,
    ParetoReportConfig,
    ParetoReporter,
    RunObservations,
    pareto_mask,
)


@pytest.mark.parametrize("directions", list(itertools.product(metrics.OptimizationDirection, repeat=2)))
def test_pareto_matches_dominance_definition(directions):
    rng = random.Random(42)
    points = [(rng.randint(-10, 10), rng.randint(-10, 10)) for _ in range(100)]
    signs = [1 if d == metrics.OptimizationDirection.MAXIMIZE else -1 for d in directions]
    scores = [(signs[0] * x, signs[1] * y) for x, y in points]
    expected = [not any(a >= x and b >= y and (a > x or b > y) for a, b in scores) for x, y in scores]
    assert pareto_mask(points, *directions) == expected
    assert pareto_mask([], *directions) == []
    assert pareto_mask([(0, 0), (0, 0)], *directions) == [True, True]
    with pytest.raises(ValueError, match="finite"):
        pareto_mask([(float("nan"), 1)], *directions)


def plot_config(cases: list[str]) -> ParetoPlotConfig:
    return ParetoPlotConfig.model_validate(
        {
            "name": "Latency vs throughput",
            "x": {"metric": "ttft", "match": {"statistic": "p99"}},
            "y": {"metric": "output_token_throughput"},
            "frontiers": [{"name": "Version A", "cases": cases}],
        }
    )


def test_configuration_and_selectors():
    config = {"enable": True, "plots": [plot_config(["a"]).model_dump()]}
    parsed = parse_reports_spec({"pareto": config})
    assert parsed is not None and isinstance(parsed["pareto"], ParetoReportConfig)
    assert plot_config(["a"]).x.resolved_direction == metrics.OptimizationDirection.MINIMIZE
    assert (
        MetricSelector(metric="ttft", direction=metrics.OptimizationDirection.MAXIMIZE).resolved_direction.value
        == "maximize"
    )
    for change in ({"metric": "missing"}, {"metric": "ttft", "direction": "up"}):
        with pytest.raises(ValidationError):
            MetricSelector.model_validate(change)
    with pytest.raises(ValidationError, match="at least one plot"):
        ParetoReportConfig(enable=True)


def observations(x: float, y: float):
    return [
        metrics.MetricObservation(metrics.TTFT, x, {"statistic": "p99"}),
        metrics.MetricObservation(metrics.OUTPUT_TOKEN_THROUGHPUT, y, {}),
    ]


def test_missing_ambiguous_and_incompatible_metrics(base_tr: TestRun):
    plot = plot_config([base_tr.name])
    frontier = plot.frontiers[0]
    run = RunObservations(base_tr, observations(1, 10))
    run.observations.append(metrics.MetricObservation(metrics.TTFT, 2, {"statistic": "mean"}))
    plot.x.match = {}
    with pytest.raises(ValueError, match="Ambiguous"):
        ParetoReporter.select_pair(run, plot, frontier)
    plot.x.match = {"statistic": "p99"}
    assert ParetoReporter.select_pair(run, plot, frontier)[0].value == 1
    run.observations = run.observations[:1]
    with pytest.raises(ValueError, match="Missing"):
        ParetoReporter.select_pair(run, plot, frontier)
    run.observations = [
        metrics.MetricObservation(metrics.TTFT, 1, {"statistic": "p99", "backend": "one"}),
        metrics.MetricObservation(metrics.OUTPUT_TOKEN_THROUGHPUT, 10, {"backend": "two"}),
    ]
    with pytest.raises(ValueError, match="incompatible"):
        ParetoReporter.select_pair(run, plot, frontier)
    frontier.match = {"backend": "one"}
    with pytest.raises(ValueError, match="Missing"):
        ParetoReporter.select_pair(run, plot, frontier)
