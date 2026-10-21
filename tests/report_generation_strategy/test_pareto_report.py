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

import pytest
import toml

import cloudai.core
import cloudai.models.scenario
import cloudai.report_generator.pareto_report as pareto_report
import cloudai.systems.slurm
import cloudai.workloads.sglang
import cloudai.workloads.vllm


@pytest.fixture(
    params=list(itertools.product(["vllm", "sglang"], ["minimize", "maximize"], ["minimize", "maximize"])),
    ids=lambda values: "-".join(values),
)
def pareto_reporter(
    slurm_system: cloudai.systems.slurm.SlurmSystem, tmp_path: Path, request: pytest.FixtureRequest
) -> pareto_report.ParetoReporter:
    workload, x_direction, y_direction = request.param
    cls = (
        cloudai.workloads.vllm.VllmTestDefinition
        if workload == "vllm"
        else cloudai.workloads.sglang.SglangTestDefinition
    )
    definition = cls.model_validate(
        {
            "name": "test",
            "description": "test",
            "test_template_name": workload,
            "cmd_args": {"docker_image_url": "example/image:latest", "model": "example/model"},
        }
    )
    executed = copy.deepcopy(definition)
    executed.bench_cmd_args.max_concurrency = 8
    setattr(executed.cmd_args.decode, "tensor_parallel_size" if workload == "vllm" else "tp_size", 4)
    for relative, x, y in [("ordinary/0", 2, 20), ("ordinary/1", 1, 10), ("ordinary/2", 2, 20), ("sweep/0/1", 3, 15)]:
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        data = {
            "num_prompts": 10,
            "completed": 10,
            "max_concurrency": 4,
            "mean_ttft_ms": x / 2,
            "median_ttft_ms": x * 0.75,
            "p99_ttft_ms": x,
            "mean_tpot_ms": 1,
            "median_tpot_ms": 2,
            "p99_tpot_ms": 3,
            "request_throughput": 1,
            "output_throughput": y,
        }
        (folder / ("vllm-bench.json" if workload == "vllm" else "sglang-bench.jsonl")).write_text(json.dumps(data))
        (folder / "vllm-bench.log").write_text(
            "============ Serving Benchmark Result ============\nSuccessful requests: 10\n"
        )
        if relative.startswith("sweep"):
            resolved = cloudai.core.TestRun("sweep", executed, 2, [], output_path=folder)
            (folder / "test-run.toml").write_text(
                toml.dumps(cloudai.models.scenario.TestRunDetails.from_test_run(resolved, "", "").model_dump())
            )
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
    sweep_definition["bench_cmd_args"]["max_concurrency"] = 8
    tp_key = "tensor_parallel_size" if definition["test_template_name"] == "vllm" else "tp_size"
    sweep_definition["cmd_args"]["decode"][tp_key] = 4
    context = {
        "Workload": definition["test_template_name"],
        "Nodes": 1,
        "Allocated GPUs": 8,
        "Serving GPUs": 1,
        "Model": "example/model",
        "Input tokens": 16,
        "Output tokens": 128,
        "Concurrency": 16,
        "Requests": 30,
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
            "context": {
                **context,
                "Nodes": 2,
                "Allocated GPUs": 16,
                "Serving GPUs": 4,
                "Concurrency": 8,
                "Decode / TP": 4,
            },
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
    plot = pareto_reporter.config.plots[0]
    expected_frontier = {
        ("minimize", "maximize"): [True, True, True, False],
        ("maximize", "maximize"): [True, False, True, True],
        ("minimize", "minimize"): [False, True, False, False],
        ("maximize", "minimize"): [False, True, False, True],
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
    chart_points = [{k: v for k, v in points[i].items() if k != "config"} for i in (1, 0, 2, 3)]
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
                [20 / 8, 10 / 8, 20 / 8, 15 / 16],
                ["Ordinary", "Ordinary", "Ordinary", "Sweep"],
                ["Frontier", "Dominated", "Frontier", "Frontier"],
                strict=True,
            )
        )
    ]
    normalized_data = [{k: v for k, v in point.items() if k != "config"} for point in normalized_points]
    expected = [
        {
            "pareto": True,
            "title": "All configurations",
            "summaries": [{"name": "All", "candidates": 4, "frontier": sum(expected_frontier), "excluded": 3}],
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
                {"name": "Sweep", "candidates": 1, "frontier": 1, "excluded": 3},
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
                        "data": [normalized_data[3]],
                        "source_color_index": 1,
                        "is_frontier": True,
                        "hidden": False,
                    },
                ],
            },
        },
        *[
            {
                "pareto": True,
                "title": title,
                "summaries": [{"name": "All", "candidates": 0, "frontier": 0, "excluded": 7}],
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
    assert pareto_reporter.build_sections(pareto_reporter.collect_runs()) == expected


def test_generate_html(pareto_reporter: pareto_report.ParetoReporter) -> None:
    pareto_reporter.generate()
    assert (pareto_reporter.results_root / "pareto.html").stat().st_size > 0
