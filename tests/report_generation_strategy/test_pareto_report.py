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

import copy
import itertools
import json
import random
from pathlib import Path

import pytest
import toml
from pydantic import ValidationError

import cloudai.metrics as metrics
from cloudai.core import TestRun, TestScenario
from cloudai.models.scenario import TestRunDetails, parse_reports_spec
from cloudai.report_generator.pareto_report import (
    MetricSelector,
    ParetoPlotConfig,
    ParetoReportConfig,
    ParetoReporter,
    RunObservations,
    pareto_mask,
)
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import SglangTestDefinition
from cloudai.workloads.vllm import VllmTestDefinition


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


def test_grouping_invariance_and_independent_frontiers(slurm_system: SlurmSystem, base_tr: TestRun):
    config = ParetoReportConfig(enable=True, plots=[plot_config(["a", "b"])])
    report = ParetoReporter(slurm_system, TestScenario("scenario", []), base_tr.output_path, config)
    runs = []
    for case, x, y in [("a", 1, 10), ("a", 2, 20), ("b", 3, 15)]:
        tr = copy.deepcopy(base_tr)
        tr.name = case
        runs.append(RunObservations(tr, observations(x, y)))
    first = report.build_sections(runs)[0]
    assert [p["status"] for p in first["points"]] == ["Frontier", "Frontier", "Dominated"]
    for run in runs:
        run.tr.name = "a"
    assert [p["status"] for p in report.build_sections(runs)[0]["points"]] == [p["status"] for p in first["points"]]
    runs[-1].tr.name = "b"
    config.plots[0].frontiers = [
        type(config.plots[0].frontiers[0])(name="A", cases=["a"]),
        type(config.plots[0].frontiers[0])(name="B", cases=["b"]),
    ]
    assert all(p["status"] == "Frontier" for p in report.build_sections(runs)[0]["points"])


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


@pytest.mark.parametrize("workload", ["vllm", "sglang"])
def test_real_workload_iterations_and_search_steps(slurm_system: SlurmSystem, tmp_path: Path, workload: str):
    cls = VllmTestDefinition if workload == "vllm" else SglangTestDefinition
    definition = cls.model_validate(
        {
            "name": "test",
            "description": "test",
            "test_template_name": workload,
            "cmd_args": {"docker_image_url": "example/image:latest"},
        }
    )
    ordinary = TestRun("ordinary", definition, 1, [], iterations=2)
    search = TestRun("search", definition.model_copy(deep=True), [1, 2], [])
    missing = TestRun("missing", definition.model_copy(deep=True), 1, [])
    for relative, x, y in [("ordinary/0", 1, 10), ("ordinary/1", 2, 20), ("search/0/1", 3, 15)]:
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        data = {
            "num_prompts": 10,
            "completed": 10,
            "max_concurrency": 4,
            "mean_ttft_ms": x,
            "median_ttft_ms": x,
            "p99_ttft_ms": x,
            "mean_tpot_ms": 1,
            "median_tpot_ms": 1,
            "p99_tpot_ms": 1,
            "request_throughput": 1,
            "output_throughput": y,
            "output_throughput_per_gpu": y,
            "input_throughput": y,
        }
        (folder / ("vllm-bench.json" if workload == "vllm" else "sglang-bench.jsonl")).write_text(json.dumps(data))
        (folder / "vllm-bench.log").write_text(
            "============ Serving Benchmark Result ============\nSuccessful requests: 10\n"
        )
        resolved = TestRun("search", definition, 2, [], output_path=folder)
        if relative.startswith("search"):
            (folder / "test-run.toml").write_text(
                toml.dumps(TestRunDetails.from_test_run(resolved, "", "").model_dump())
            )
    (tmp_path / "search/0/2").mkdir()
    (tmp_path / "search/0/2/test-run.toml").write_text("invalid = [")
    (tmp_path / "search/0/3").mkdir()
    (tmp_path / "search/controller").mkdir()
    config = ParetoReportConfig(enable=True, plots=[plot_config(["ordinary", "search", "missing"])])
    report = ParetoReporter(slurm_system, TestScenario("example", [ordinary, search, missing]), tmp_path, config)
    collected = report.collect_runs()
    section = report.build_sections(collected)[0]
    assert len(section["points"]) == 3
    assert len(section["excluded"]) == 3
    assert section["summaries"][0]["frontier"] == 2
    assert "num_nodes = 2" in section["points"][-1]["config"]
    report.generate()
    html = (tmp_path / "pareto.html").read_text()
    assert '"type": "scatter"' in html
    assert "Executed configuration" in html
    assert "Missing test-run.toml" in html
    config.plots[0].frontiers[0].cases.append("unknown")
    with pytest.raises(ValueError, match="Unknown Pareto case"):
        report.collect_runs()


def test_empty_report_and_html_escaping(slurm_system: SlurmSystem, base_tr: TestRun, tmp_path: Path, monkeypatch):
    config = ParetoReportConfig(enable=True, plots=[plot_config([base_tr.name])])
    config.plots[0].name = "<script>alert(1)</script>"
    report = ParetoReporter(slurm_system, TestScenario("test", [base_tr]), tmp_path, config)
    monkeypatch.setattr(report, "collect_runs", lambda: [RunObservations(base_tr, error="<script>bad</script>")])
    report.generate()
    html = (tmp_path / "pareto.html").read_text()
    assert "No valid points" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
