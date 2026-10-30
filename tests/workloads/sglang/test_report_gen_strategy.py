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

import json
from pathlib import Path
from typing import cast

import pytest

import cloudai.core
import cloudai.metrics
import cloudai.report_generator.inference_report as inference_report
import cloudai.systems.slurm
import cloudai.util
from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import (
    SGLangBenchReport,
    SGLangBenchReportGenerationStrategy,
    SglangCmdArgs,
    SglangSemanticEvalCmdArgs,
    SglangTestDefinition,
)
from cloudai.workloads.sglang.sglang import (
    SGLANG_BENCH_JSONL_FILE,
    SGLANG_SEMANTIC_EVAL_LOG_FILE,
    parse_sglang_semantic_accuracy,
)

BENCH_RECORD = {
    "num_prompts": 30,
    "completed": 30,
    "request_throughput": 2400.0,
    "output_throughput": 4800.0,
    "max_concurrency": 16,
    "mean_ttft_ms": 120.0,
    "median_ttft_ms": 100.0,
    "p99_ttft_ms": 200.0,
    "mean_tpot_ms": 12.0,
    "median_tpot_ms": 10.0,
    "p99_tpot_ms": 20.0,
}


@pytest.fixture
def sglang_tr(tmp_path: Path) -> TestRun:
    tdef = SglangTestDefinition(
        name="sglang_test",
        description="SGLang benchmark",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev"),
    )
    tr = TestRun(name="sglang", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    (tr.output_path / SGLANG_BENCH_JSONL_FILE).write_text(
        json.dumps(BENCH_RECORD) + "\n",
        encoding="utf-8",
    )
    return tr


def test_sglang_can_handle_directory(slurm_system: SlurmSystem, sglang_tr: TestRun) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    assert strategy.can_handle_directory() is True


def test_parse_sglang_bench_output_missing_file(tmp_path: Path) -> None:
    assert SglangTestDefinition.parse_bench_output(tmp_path / SGLANG_BENCH_JSONL_FILE) is None


def test_parse_sglang_bench_output_skips_invalid_lines(tmp_path: Path) -> None:
    report_path = tmp_path / SGLANG_BENCH_JSONL_FILE
    report_path.write_text(
        "{invalid}\n" + json.dumps(BENCH_RECORD) + "\n",
        encoding="utf-8",
    )

    parsed = SglangTestDefinition.parse_bench_output(report_path)

    assert parsed == SGLangBenchReport.model_validate_json(report_path.read_text().splitlines()[1])


def test_parse_sglang_bench_output_returns_none_for_unsuccessful_record(tmp_path: Path) -> None:
    report_path = tmp_path / SGLANG_BENCH_JSONL_FILE
    report_path.write_text(
        json.dumps({**BENCH_RECORD, "completed": 0, "request_throughput": 0.0}) + "\n",
        encoding="utf-8",
    )

    assert SglangTestDefinition.parse_bench_output(report_path) is None


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("default", 2400.0),
        ("throughput", 2400.0),
        ("tps-per-user", 150.0),
        ("tps-per-gpu", 300.0),
    ],
)
def test_sglang_metrics(slurm_system: SlurmSystem, sglang_tr: TestRun, metric: str, expected: float) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    assert strategy.get_metric(metric) == expected


@pytest.mark.parametrize("metric", ["tps_per_user", "tps_per_gpu", "bw", "nonexistent"])
def test_sglang_invalid_metric_returns_error(slurm_system: SlurmSystem, sglang_tr: TestRun, metric: str) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    assert strategy.get_metric(metric) == METRIC_ERROR


def test_sglang_metric_returns_error_when_report_cannot_be_parsed(
    slurm_system: SlurmSystem, sglang_tr: TestRun
) -> None:
    (sglang_tr.output_path / SGLANG_BENCH_JSONL_FILE).write_text("{invalid}\n", encoding="utf-8")

    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)

    assert strategy.get_metric("throughput") == METRIC_ERROR


def test_sglang_tps_per_gpu(slurm_system: SlurmSystem, sglang_tr: TestRun) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    strategy.used_gpus_count = lambda: 4

    metric = strategy.get_metric("tps-per-gpu")

    assert metric == 600.0


def test_sglang_accuracy_metric(slurm_system: SlurmSystem, sglang_tr: TestRun):
    sglang_test = cast(SglangTestDefinition, sglang_tr.test)
    sglang_test.semantic_eval_cmd_args = SglangSemanticEvalCmdArgs()
    (sglang_tr.output_path / SGLANG_SEMANTIC_EVAL_LOG_FILE).write_text("Score: 0.945\n", encoding="utf-8")

    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)

    assert strategy.get_metric("accuracy") == 0.945


def test_parse_sglang_semantic_accuracy_from_score(tmp_path: Path):
    log_path = tmp_path / "score.log"
    log_path.write_text("Total latency: 1.000 s\nScore: 0.812\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(log_path) == 0.812


def test_parse_sglang_semantic_accuracy_from_legacy_accuracy(tmp_path: Path) -> None:
    log_path = tmp_path / "accuracy.log"
    log_path.write_text("Accuracy: 0.945\nInvalid: 0.000\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(log_path) == 0.945


def test_parse_sglang_semantic_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    log_path = tmp_path / "invalid.log"
    log_path.write_text("no score here\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(tmp_path / "missing.log") is None
    assert parse_sglang_semantic_accuracy(log_path) is None


def test_sglang_tps_per_user__concurrency_is_zero() -> None:
    bench_report = SGLangBenchReport.model_validate({**BENCH_RECORD, "max_concurrency": 0})
    assert bench_report.tps_per_user is None


def test_sglang_parses_num_prompts_from_input_lens(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = SglangTestDefinition(
        name="sglang_test_missing_num_prompts",
        description="SGLang benchmark",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev"),
    )
    tr = TestRun(name="sglang", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    (tr.output_path / SGLANG_BENCH_JSONL_FILE).write_text(
        json.dumps({k: v for k, v in BENCH_RECORD.items() if k != "num_prompts"} | {"input_lens": [16] * 30}) + "\n",
        encoding="utf-8",
    )

    strategy = SGLangBenchReportGenerationStrategy(slurm_system, tr)
    assert strategy.can_handle_directory() is True
    assert strategy.get_metric("throughput") == 2400.0


INFERENCE_DEFINITION = {
    "name": "sglang",
    "description": "Inference report",
    "test_template_name": "sglang",
    "cmd_args": {
        "docker_image_url": "example/image:latest",
        "model": "example/model",
        "decode": {
            "tp_size": 2,
            "pp_size": 2,
            "dp_size": 1,
            "enable_expert_parallel": True,
            "dtype": "float16",
            "quantization": "fp8",
        },
    },
    "bench_cmd_args": {
        "num_prompts": 20,
        "max_concurrency": 8,
        "request_rate": 12.5,
        "random_input": 128,
        "random_output": 64,
    },
}
INFERENCE_PREFILL = {
    "tp_size": 2,
    "pp_size": 1,
    "dp_size": 1,
    "enable_expert_parallel": True,
    "dtype": "float16",
    "quantization": "fp8",
}


@pytest.mark.parametrize(
    "overrides,nodes,role_nodes,serving_gpus,shared_nodes",
    [
        pytest.param({}, 2, 2, 4, None, id="combined"),
        pytest.param(
            {"cmd_args": {"prefill": {**INFERENCE_PREFILL, "num_nodes": 2}, "decode": {"num_nodes": 2}}},
            4,
            2,
            6,
            False,
            id="separate-nodes",
        ),
        pytest.param({"cmd_args": {"prefill": INFERENCE_PREFILL}}, 1, 1, 6, True, id="shared-node"),
        pytest.param({"cmd_args": {"prefill": INFERENCE_PREFILL}}, 4, None, 6, None, id="unresolved-topology"),
    ],
)
def test_inference_reports(
    slurm_system: SlurmSystem,
    tmp_path: Path,
    overrides: dict,
    nodes: int,
    role_nodes: int | None,
    serving_gpus: int,
    shared_nodes: bool | None,
) -> None:
    definition = cloudai.util.deep_merge(INFERENCE_DEFINITION, overrides)
    test = SglangTestDefinition.model_validate(definition)
    tr = TestRun("sglang", test, nodes, [], output_path=tmp_path)
    (tmp_path / "sglang-bench.jsonl").write_text(
        json.dumps(
            {
                "num_prompts": 20,
                "completed": 20,
                "max_concurrency": 8,
                "request_throughput": 10,
                "output_throughput": 1000,
                "mean_ttft_ms": 10,
                "median_ttft_ms": 11,
                "p99_ttft_ms": 12,
                "mean_tpot_ms": 1,
                "median_tpot_ms": 2,
                "p99_tpot_ms": 3,
            }
        )
    )
    role: dict = dict(
        num_nodes=role_nodes,
        num_instances=1 if role_nodes else None,
        nodes_per_instance=role_nodes,
        instances_per_node=1 if role_nodes == 1 else None,
        tensor_parallel_size=2,
        data_parallel_size=1,
        expert_parallel=True,
        precision="float16",
        quantization="fp8",
    )
    metrics = [
        (cloudai.metrics.REQUEST_THROUGHPUT, 10, {}),
        (cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT, 1000, {}),
        (cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT_PER_ALLOCATED_GPU, 1000 / (nodes * 8), {}),
        (cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT_PER_SERVING_GPU, 1000 / serving_gpus, {}),
    ]
    metrics.extend(
        (metric, value, {"statistic": statistic})
        for metric, values in ((cloudai.metrics.TTFT, (10, 11, 12)), (cloudai.metrics.TPOT, (1, 2, 3)))
        for statistic, value in zip(("mean", "median", "p99"), values, strict=True)
    )
    expected = [
        inference_report.InferenceReport(
            config=inference_report.InferenceConfig(
                model="example/model",
                num_nodes=nodes,
                backend="sglang",
                benchmark="sglang",
                container_image="example/image:latest",
                gpus_per_node=8,
                serving_gpu_count=serving_gpus,
                input_tokens=128,
                output_tokens=64,
                concurrency=8,
                request_count=20,
                request_rate=12.5,
                prefill=inference_report.InferenceRoleConfig(**role, pipeline_parallel_size=1)
                if "prefill" in definition["cmd_args"]
                else None,
                decode=inference_report.InferenceRoleConfig(**role, pipeline_parallel_size=2),
                shared_nodes=shared_nodes,
            ),
            observations=[
                cloudai.metrics.MetricObservation(metric, value, dimensions) for metric, value, dimensions in metrics
            ],
        )
    ]
    assert test.inference_reports(slurm_system, tr) == expected
    assert test.metric_observations(slurm_system, tr) == expected[0].observations
