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
from pathlib import Path

import pydantic
import pytest
import toml

import cloudai.core
import cloudai.metrics
import cloudai.models.scenario
import cloudai.models.workload
import cloudai.report_generator.inference_report as inference_report
import cloudai.report_generator.pareto_report as pareto_report
import cloudai.systems.slurm


class DummyArgs(pydantic.BaseModel):
    """Input configuration restored from a saved sweep step."""

    concurrency: int = 16


class DummyInferenceDefinition(cloudai.models.workload.TestDefinition, inference_report.InferenceReportProvider):
    """Supply independent reports without a benchmark parser or serving backend."""

    cmd_args: DummyArgs

    def inference_reports(
        self, system: cloudai.core.System, tr: cloudai.core.TestRun
    ) -> list[inference_report.InferenceReport]:
        gpus_per_node = getattr(system, "gpus_per_node", None)
        assert gpus_per_node is not None
        reports = []
        for measurement in json.loads((tr.output_path / "measurements.json").read_text()):
            dimensions = {"phase": measurement["phase"]} if "phase" in measurement else {}
            observations = [
                cloudai.metrics.MetricObservation(metric, value * factor, {**dimensions, "statistic": statistic})
                for metric, value, factors in (
                    (cloudai.metrics.TTFT, measurement["x"], (0.5, 0.75, 1)),
                    (cloudai.metrics.TPOT, measurement["tpot"], (1, 2, 3)),
                )
                for statistic, factor in zip(("mean", "median", "p99"), factors, strict=True)
            ]
            observations.extend(
                [
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT, measurement["y"], dimensions
                    ),
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT_PER_ALLOCATED_GPU,
                        measurement["y"] / (tr.nnodes * gpus_per_node),
                        dimensions,
                    ),
                ]
            )
            reports.append(
                inference_report.InferenceReport(
                    config=inference_report.InferenceConfig(
                        model="example/model",
                        num_nodes=tr.nnodes,
                        gpus_per_node=gpus_per_node,
                        concurrency=measurement.get("concurrency", self.cmd_args.concurrency),
                        backend="example-backend",
                        container_image="example/image:latest",
                        serving_gpu_count=4,
                        input_tokens=128,
                        output_tokens=64,
                        users=8,
                        request_count=32,
                        request_rate=0,
                        shared_nodes=False,
                        prefill=inference_report.InferenceRoleConfig(num_nodes=1) if dimensions else None,
                        decode=inference_report.InferenceRoleConfig(
                            model="example/role-model",
                            num_nodes=1,
                            num_instances=1,
                            nodes_per_instance=1,
                            instances_per_node=1,
                            gpu_ids=[0, 1],
                            tensor_parallel_size=2,
                            pipeline_parallel_size=1,
                            data_parallel_size=1,
                            expert_parallel=False,
                            precision="bfloat16",
                            quantization="fp8",
                        ),
                    ),
                    observations=observations,
                )
            )
        return reports


@pytest.fixture(params=list(itertools.product(["minimize", "maximize"], repeat=2)), ids=lambda values: "-".join(values))
def pareto_reporter(
    slurm_system: cloudai.systems.slurm.SlurmSystem,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> pareto_report.ParetoReporter:
    monkeypatch.setitem(cloudai.core.Registry().test_definitions_map, "dummy", DummyInferenceDefinition)
    monkeypatch.setitem(
        cloudai.metrics.MetricCatalog._dimensions,
        "phase",
        cloudai.metrics.DimensionDefinition("phase", "Phase", str),
    )
    x_direction, y_direction = request.param
    definition = DummyInferenceDefinition(
        name="test", description="test", test_template_name="dummy", cmd_args=DummyArgs()
    )
    executed = copy.deepcopy(definition)
    executed.cmd_args.concurrency = 8
    for relative, x, y in [("ordinary/0", 2, 20), ("ordinary/1", 1, 10), ("ordinary/2", 2, 20), ("sweep/0/1", 3, 15)]:
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        measurements = [{"x": x, "y": y, "tpot": 1}]
        if relative.startswith("sweep"):
            measurements = [
                {"x": x, "y": y, "tpot": 1, "phase": "low"},
                {"x": 4, "y": 30, "tpot": 2, "phase": "high", "concurrency": 32},
            ]
            resolved = cloudai.core.TestRun("sweep", executed, 2, [], output_path=folder)
            (folder / "test-run.toml").write_text(
                toml.dumps(cloudai.models.scenario.TestRunDetails.from_test_run(resolved, "", "").model_dump())
            )
        (folder / "measurements.json").write_text(json.dumps(measurements))
    for relative in ("sweep/0/2", "sweep/controller", "empty-sweep/0/controller"):
        (tmp_path / relative).mkdir(parents=True)
    plot = {
        "name": "All configurations",
        "x": {"metric": "ttft", "match": {"statistic": "p99"}, "direction": x_direction},
        "y": {"metric": "output_token_throughput", "direction": y_direction},
        "frontiers": [{"name": "All"}],
    }
    config = pareto_report.ParetoReportConfig.model_validate(
        {
            "enable": True,
            "plots": [
                plot,
                {
                    **plot,
                    "name": "Per allocated GPU",
                    "x": {"metric": "tpot", "match": {"statistic": "mean"}, "direction": "maximize"},
                    "y": {"metric": "output_token_throughput_per_allocated_gpu"},
                    "frontiers": [
                        {"name": "Ordinary", "cases": ["ordinary"]},
                        {"name": "Sweep", "cases": ["sweep", "missing", "empty-sweep"]},
                    ],
                },
                {
                    **plot,
                    "name": "Missing accuracy",
                    "y": {"metric": "accuracy"},
                    "frontiers": [{"name": "All", "cases": []}],
                },
                {**plot, "name": "Ambiguous latency", "x": {"metric": "ttft"}},
                {
                    **plot,
                    "name": "Selected phase",
                    "frontiers": [
                        {"name": "High", "match": {"phase": "high"}},
                        {"name": "Missing", "match": {"phase": "missing"}},
                    ],
                },
            ],
        }
    )
    scenario = cloudai.core.TestScenario(
        name="example",
        test_runs=[
            cloudai.core.TestRun("ordinary", definition, 1, [], iterations=3),
            cloudai.core.TestRun("sweep", definition, [1, 2], []),
            cloudai.core.TestRun("missing", definition, 1, []),
            cloudai.core.TestRun("empty-sweep", definition, [1, 2], []),
        ],
    )
    return pareto_report.ParetoReporter(slurm_system, scenario, tmp_path, config)


def test_build_sections(pareto_reporter: pareto_report.ParetoReporter) -> None:
    definition = pareto_reporter.test_scenario.test_runs[0].test.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    sweep_definition = copy.deepcopy(definition)
    sweep_definition["cmd_args"]["concurrency"] = 8
    context = {
        "Workload": "dummy",
        "Model": "example/model",
        "Nodes": 1,
        "Allocated GPUs": 8,
        "Concurrency": 16,
        "Backend": "example-backend",
        "Container image": "example/image:latest",
        "Serving GPUs": 4,
        "Input tokens": 128,
        "Output tokens": 64,
        "Users": 8,
        "Requests": 32,
        "Request rate": 0,
        "Prefill and decode share nodes": False,
        "Serving / Model": "example/role-model",
        "Serving / Nodes": 1,
        "Serving / Instances": 1,
        "Serving / Nodes per instance": 1,
        "Serving / Instances per node": 1,
        "Serving / GPU IDs": [0, 1],
        "Serving / TP": 2,
        "Serving / PP": 1,
        "Serving / DP": 1,
        "Serving / Expert parallel": False,
        "Serving / Precision": "bfloat16",
        "Serving / Quantization": "fp8",
    }
    sweep_context = {
        **{key.replace("Serving / ", "Decode / "): value for key, value in context.items()},
        "Nodes": 2,
        "Allocated GPUs": 16,
        "Concurrency": 8,
        "Prefill / Nodes": 1,
    }
    points = [
        {
            "x": 2,
            "y": 20,
            "run": "ordinary · iteration=0 · step=0",
            "group": "All",
            "context": context,
            "dimensions": "x.statistic=p99",
            "config": toml.dumps({"num_nodes": 1, "test": definition}),
            "artifact": "ordinary/0",
            "status": "Frontier",
            "id": "pareto-0-point-0",
        },
        {
            "x": 1,
            "y": 10,
            "run": "ordinary · iteration=1 · step=0",
            "group": "All",
            "context": context,
            "dimensions": "x.statistic=p99",
            "config": toml.dumps({"num_nodes": 1, "test": definition}),
            "artifact": "ordinary/1",
            "status": "Frontier",
            "id": "pareto-0-point-1",
        },
        {
            "x": 3,
            "y": 15,
            "run": "sweep · iteration=0 · step=1",
            "group": "All",
            "context": sweep_context,
            "dimensions": "x.statistic=p99",
            "config": toml.dumps({"num_nodes": 2, "test": sweep_definition}),
            "artifact": "sweep/0/1",
            "status": "Dominated",
            "id": "pareto-0-point-3",
        },
    ]
    points.insert(
        2, {**points[0], "run": "ordinary · iteration=2 · step=0", "artifact": "ordinary/2", "id": "pareto-0-point-2"}
    )
    sweep_point = points.pop()
    for index, (phase, concurrency, x, y) in enumerate([("low", 8, 3, 15), ("high", 32, 4, 30)], start=3):
        point = {
            **sweep_point,
            "x": x,
            "y": y,
            "run": "sweep · iteration=0 · step=1",
            "context": {**sweep_point["context"], "Concurrency": concurrency},
            "dimensions": f"x.phase={phase}, x.statistic=p99, y.phase={phase}",
            "id": f"pareto-0-point-{index}",
        }
        points.append(point)
    plot = pareto_reporter.config.plots[0]
    expected_frontier = {
        ("minimize", "maximize"): [True, True, True, False, True],
        ("maximize", "maximize"): [False, False, False, False, True],
        ("minimize", "minimize"): [False, True, False, False, False],
        ("maximize", "minimize"): [False, True, False, True, True],
    }[plot.x.resolved_direction.value, plot.y.resolved_direction.value]
    for point, on_frontier in zip(points, expected_frontier, strict=True):
        point["status"] = "Frontier" if on_frontier else "Dominated"
    excluded = [
        {"group": "All", "run": "sweep · iteration=0 · step=2", "reason": "Missing test-run.toml for sweep step"},
        {"group": "All", "run": "missing · iteration=0 · step=0", "reason": "No execution results found"},
        {"group": "All", "run": "empty-sweep · iteration=0 · step=0", "reason": "No sweep steps found in iteration 0"},
    ]
    chart = {
        "id": "pareto-0",
        "type": "scatter",
        "pareto": True,
        "x_axis_type": "linear",
        "y_axis_type": "linear",
        "x_axis_label": "P99 Time to first token (ms)",
        "y_axis_label": "Output token throughput (tokens/s)",
    }
    chart_points = [{k: v for k, v in points[i].items() if k != "config"} for i in (1, 0, 2, 3, 4)]
    normalized_points = [
        {
            **point,
            "x": 1,
            "y": y,
            "group": group,
            "status": status,
            "dimensions": "x.statistic=mean",
            "id": f"pareto-1-point-{i}",
        }
        for i, (point, y, group, status) in enumerate(
            zip(
                points,
                [20 / 8, 10 / 8, 20 / 8, 15 / 16, 30 / 16],
                ["Ordinary", "Ordinary", "Ordinary", "Sweep", "Sweep"],
                ["Frontier", "Dominated", "Frontier", "Dominated", "Frontier"],
                strict=True,
            )
        )
    ]
    normalized_points[3]["dimensions"] = "x.phase=low, x.statistic=mean, y.phase=low"
    normalized_points[4]["dimensions"] = "x.phase=high, x.statistic=mean, y.phase=high"
    normalized_points[4]["x"] = 2
    normalized_data = [{k: v for k, v in point.items() if k != "config"} for point in normalized_points]
    expected = [
        {
            "pareto": True,
            "title": "All configurations",
            "summaries": [
                {"name": "All", "candidates": len(points), "frontier": sum(expected_frontier), "excluded": 3}
            ],
            "points": points,
            "excluded": excluded,
            "chart": {
                **chart,
                "datasets": [
                    {
                        "label": "All",
                        "data": [point for point in chart_points if point["status"] == "Frontier"],
                        "source_color_index": 0,
                        "is_frontier": True,
                        "hidden": False,
                    },
                    {
                        "label": "All · dominated",
                        "data": [point for point in chart_points if point["status"] == "Dominated"],
                        "source_color_index": 0,
                        "is_frontier": False,
                        "hidden": True,
                    },
                ],
            },
        },
        {
            "pareto": True,
            "title": "Per allocated GPU",
            "summaries": [
                {"name": "Ordinary", "candidates": 3, "frontier": 2, "excluded": 0},
                {"name": "Sweep", "candidates": 2, "frontier": 1, "excluded": 3},
            ],
            "points": normalized_points,
            "excluded": [{**item, "group": "Sweep"} for item in excluded],
            "chart": {
                **chart,
                "id": "pareto-1",
                "x_axis_label": "MEAN Time per output token (ms)",
                "y_axis_label": "Output throughput per allocated GPU (tokens/s/GPU)",
                "datasets": [
                    {
                        "label": "Ordinary",
                        "data": [normalized_data[0], normalized_data[2]],
                        "source_color_index": 0,
                        "is_frontier": True,
                        "hidden": False,
                    },
                    {
                        "label": "Ordinary · dominated",
                        "data": [normalized_data[1]],
                        "source_color_index": 0,
                        "is_frontier": False,
                        "hidden": True,
                    },
                    {
                        "label": "Sweep",
                        "data": [normalized_data[4]],
                        "source_color_index": 1,
                        "is_frontier": True,
                        "hidden": False,
                    },
                    {
                        "label": "Sweep · dominated",
                        "data": [normalized_data[3]],
                        "source_color_index": 1,
                        "is_frontier": False,
                        "hidden": True,
                    },
                ],
            },
        },
        *[
            {
                "pareto": True,
                "title": title,
                "summaries": [{"name": "All", "candidates": 0, "frontier": 0, "excluded": len(points) + 3}],
                "points": [],
                "excluded": [{"group": "All", "run": point["run"], "reason": reason} for point in points] + excluded,
                "chart": {
                    **chart,
                    "id": f"pareto-{index}",
                    "x_axis_label": x_label,
                    "y_axis_label": y_label,
                    "datasets": [],
                },
            }
            for index, title, reason, x_label, y_label in [
                (
                    2,
                    "Missing accuracy",
                    "Missing metric 'accuracy' matching {}",
                    "P99 Time to first token (ms)",
                    "Accuracy (ratio)",
                ),
                (
                    3,
                    "Ambiguous latency",
                    "Ambiguous metric 'ttft': 3 observations; specify match dimensions",
                    "Time to first token (ms)",
                    "Output token throughput (tokens/s)",
                ),
            ]
        ],
    ]
    selected = {**points[-1], "group": "High", "status": "Frontier", "id": "pareto-4-point-0"}
    expected.append(
        {
            "pareto": True,
            "title": "Selected phase",
            "summaries": [
                {"name": "High", "candidates": 1, "frontier": 1, "excluded": 3},
                {"name": "Missing", "candidates": 0, "frontier": 0, "excluded": 3},
            ],
            "points": [selected],
            "excluded": [{**item, "group": group} for group in ("High", "Missing") for item in excluded],
            "chart": {
                **chart,
                "id": "pareto-4",
                "datasets": [
                    {
                        "label": "High",
                        "data": [{key: value for key, value in selected.items() if key != "config"}],
                        "source_color_index": 0,
                        "is_frontier": True,
                        "hidden": False,
                    }
                ],
            },
        }
    )
    runs = pareto_reporter.collect_runs()
    assert pareto_reporter.build_sections(runs) == expected
    _assert_unmatched_frontiers_keep_errors(pareto_reporter, runs, expected)


def _assert_unmatched_frontiers_keep_errors(pareto_reporter, runs, expected) -> None:
    for plot in pareto_reporter.config.plots:
        for frontier in plot.frontiers:
            frontier.match = {"benchmark": "not-present"}
    filtered_expected = copy.deepcopy(expected)
    for section in filtered_expected:
        section["points"] = []
        section["chart"]["datasets"] = []
        section["excluded"] = [
            item
            for item in section["excluded"]
            if item["reason"]
            in {
                "Missing test-run.toml for sweep step",
                "No execution results found",
                "No sweep steps found in iteration 0",
            }
        ]
        for summary in section["summaries"]:
            summary["candidates"] = 0
            summary["frontier"] = 0
            summary["excluded"] = sum(item["group"] == summary["name"] for item in section["excluded"])
    assert pareto_reporter.build_sections(runs) == filtered_expected


def test_generate_html(pareto_reporter: pareto_report.ParetoReporter) -> None:
    pareto_reporter.generate()
    assert (pareto_reporter.results_root / "pareto.html").stat().st_size > 0
