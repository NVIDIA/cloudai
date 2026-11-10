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

import cloudai.metrics
import cloudai.report_generator.inference_report as inference_report
import cloudai.util
from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.vllm import (
    VLLMBenchReport,
    VLLMBenchReportGenerationStrategy,
    VllmCmdArgs,
    VllmSemanticEvalCmdArgs,
    VllmTestDefinition,
)
from cloudai.workloads.vllm.vllm import (
    VLLM_BENCH_JSON_FILE,
    VLLM_GSM8K_JSON_FILE,
    VLLM_SEMANTIC_EVAL_LOG_FILE,
    VLLM_SERVE_LOG_FILE,
    VllmServeStats,
    parse_vllm_semantic_accuracy,
    parse_vllm_serve_stats,
)

BENCH_DATA = VLLMBenchReport(
    num_prompts=30,
    completed=30,
    mean_ttft_ms=120.0,
    median_ttft_ms=118.0,
    p99_ttft_ms=200.0,
    mean_tpot_ms=12.0,
    median_tpot_ms=11.0,
    p99_tpot_ms=19.0,
    output_throughput=2400.0,
    request_throughput=10.0,
    max_concurrency=16,
)


@pytest.fixture
def vllm_tr(tmp_path: Path) -> TestRun:
    tdef = VllmTestDefinition(
        name="vllm_test",
        description="vLLM benchmark",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest"),
    )
    tr = TestRun(name="vllm", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    (tr.output_path / VLLM_BENCH_JSON_FILE).write_text(BENCH_DATA.model_dump_json(), encoding="utf-8")
    return tr


def test_vllm_can_handle_directory(slurm_system: SlurmSystem, vllm_tr: TestRun) -> None:
    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)
    assert strategy.can_handle_directory() is True


def test_parse_vllm_bench_output_missing_file(tmp_path: Path) -> None:
    assert VllmTestDefinition.parse_bench_output(tmp_path / VLLM_BENCH_JSON_FILE) is None


def test_parse_vllm_bench_output_invalid_json(tmp_path: Path) -> None:
    report_path = tmp_path / VLLM_BENCH_JSON_FILE
    report_path.write_text("{invalid", encoding="utf-8")

    assert VllmTestDefinition.parse_bench_output(report_path) is None


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("default", BENCH_DATA.throughput),
        ("throughput", BENCH_DATA.throughput),
        ("tps-per-user", BENCH_DATA.tps_per_user),
        ("tps-per-gpu", BENCH_DATA.throughput / 8),
    ],
)
def test_vllm_metrics(slurm_system: SlurmSystem, vllm_tr: TestRun, metric: str, expected: float) -> None:
    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)
    assert strategy.get_metric(metric) == expected


def test_vllm_tps_per_user__concurrency_is_zero() -> None:
    bench_report = BENCH_DATA.model_copy(update={"max_concurrency": 0})
    assert bench_report.tps_per_user is None


@pytest.mark.parametrize("metric", ["tps_per_user", "tps_per_gpu", "bw", "nonexistent"])
def test_vllm_invalid_metric_returns_error(slurm_system: SlurmSystem, vllm_tr: TestRun, metric: str) -> None:
    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)
    assert strategy.get_metric(metric) == METRIC_ERROR


def test_vllm_metric_returns_error_when_report_cannot_be_parsed(slurm_system: SlurmSystem, vllm_tr: TestRun) -> None:
    (vllm_tr.output_path / VLLM_BENCH_JSON_FILE).write_text("{invalid", encoding="utf-8")

    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)

    assert strategy.get_metric("throughput") == METRIC_ERROR


@pytest.mark.parametrize("ngpus", [1, 2, 4, 8])
def test_vllm_tps_per_gpu(slurm_system: SlurmSystem, vllm_tr: TestRun, ngpus: int) -> None:
    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)
    strategy.used_gpus_count = lambda: ngpus

    metric = strategy.get_metric("tps-per-gpu")

    assert metric == BENCH_DATA.throughput / ngpus


def test_vllm_accuracy_metric(slurm_system: SlurmSystem, vllm_tr: TestRun) -> None:
    vllm_test = cast(VllmTestDefinition, vllm_tr.test)
    vllm_test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs()
    (vllm_tr.output_path / VLLM_GSM8K_JSON_FILE).write_text('{"accuracy": 0.875}', encoding="utf-8")

    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)

    assert strategy.get_metric("accuracy") == 0.875


def test_parse_vllm_semantic_accuracy_from_json(tmp_path: Path) -> None:
    (tmp_path / VLLM_GSM8K_JSON_FILE).write_text('{"accuracy": 0.91}', encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) == 0.91


def test_parse_vllm_semantic_accuracy_falls_back_to_log(tmp_path: Path) -> None:
    (tmp_path / VLLM_GSM8K_JSON_FILE).write_text("{invalid", encoding="utf-8")
    (tmp_path / VLLM_SEMANTIC_EVAL_LOG_FILE).write_text("Accuracy: 0.742\n", encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) == 0.742


def test_parse_vllm_semantic_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    (tmp_path / VLLM_SEMANTIC_EVAL_LOG_FILE).write_text("no accuracy here\n", encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) is None


INFERENCE_DEFINITION = {
    "name": "vllm",
    "description": "Inference report",
    "test_template_name": "vllm",
    "cmd_args": {
        "docker_image_url": "example/image:latest",
        "model": "example/model",
        "decode": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
            "dtype": "float16",
            "quantization": "fp8",
        },
    },
    "bench_cmd_args": {
        "num_prompts": 20,
        "max_concurrency": 8,
        "request_rate": 12.5,
        "random_input_len": 128,
        "random_output_len": 64,
    },
}
INFERENCE_PREFILL = {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
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
    test = VllmTestDefinition.model_validate(definition)
    tr = TestRun("vllm", test, nodes, [], output_path=tmp_path)
    (tmp_path / "vllm-bench.json").write_text(
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
                backend="vllm",
                benchmark="vllm",
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


STATS_PREFIX = "(APIServer pid=101) INFO 11-09 15:24:46 [loggers.py:274] Engine 000: "
SERVE_LOG = "\n".join(
    [
        "(EngineCore_DP0 pid=202) INFO 11-09 15:23:10 [kv_cache_utils.py:1169] GPU KV cache size: 401,856 tokens, "
        "Maximum concurrency for 40,960 tokens per request: 9.81x",
        STATS_PREFIX + "Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, "
        "Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%",
        STATS_PREFIX + "Avg prompt throughput: 512.0 tokens/s, Avg generation throughput: 1200.5 tokens/s, "
        "Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 20.0%, Prefix cache hit rate: 0.0%",
        STATS_PREFIX + "Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2400.0 tokens/s, "
        "Running: 16 reqs, Waiting: 4 reqs, Finished: 2 reqs, Preemptions: 3, GPU KV cache usage: 95.5%",
        STATS_PREFIX + "Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2100.0 tokens/s, "
        "Running: 12 reqs, Waiting: 1 reqs, Finished: 5 reqs, GPU KV cache usage: 60.5%",
        "(EngineCore_DP0 pid=202) INFO 11-09 15:25:01 [loggers.py:206] Engine 000: Iteration(9): 1 context requests, "
        "16 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 3.10 ms, "
        "GPU KV cache usage: 99.0%",
        'INFO:     127.0.0.1:41234 - "POST /v1/completions HTTP/1.1" 200 OK',
    ]
)


def test_parse_vllm_serve_stats(tmp_path: Path) -> None:
    log_path = tmp_path / VLLM_SERVE_LOG_FILE
    log_path.write_text(SERVE_LOG, encoding="utf-8")

    assert parse_vllm_serve_stats(log_path) == VllmServeStats(
        kv_cache_usage=[20.0, 95.5, 60.5], waiting_requests=[0, 4, 1], kv_cache_capacity=401856
    )


def test_parse_vllm_serve_stats_legacy_capacity_line(tmp_path: Path) -> None:
    log_path = tmp_path / VLLM_SERVE_LOG_FILE
    log_path.write_text(
        "INFO 05-01 12:00:00 [kv_cache_utils.py:634] GPU KV cache size: 1,024 tokens\n"
        "INFO 05-01 12:00:00 [kv_cache_utils.py:637] Maximum concurrency for 4,096 tokens per request: 0.25x\n",
        encoding="utf-8",
    )

    assert parse_vllm_serve_stats(log_path) == VllmServeStats(
        kv_cache_usage=[], waiting_requests=[], kv_cache_capacity=1024
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("INFO Starting vLLM API server\nGPU KV cache usage: abc%\n", id="no-stats"),
        pytest.param(
            STATS_PREFIX + "Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, "
            "GPU KV cache usage: 0.0%",
            id="idle-only",
        ),
    ],
)
def test_parse_vllm_serve_stats_without_values(tmp_path: Path, content: str | None) -> None:
    log_path = tmp_path / VLLM_SERVE_LOG_FILE
    if content is not None:
        log_path.write_text(content, encoding="utf-8")

    assert parse_vllm_serve_stats(log_path) is None


def test_parse_vllm_serve_stats_ignores_inconsistent_capacities(tmp_path: Path) -> None:
    log_path = tmp_path / VLLM_SERVE_LOG_FILE
    log_path.write_text("GPU KV cache size: 1,024 tokens\nGPU KV cache size: 2,048 tokens\n", encoding="utf-8")

    assert parse_vllm_serve_stats(log_path) is None


def test_vllm_serve_stats_observations() -> None:
    stats = VllmServeStats(kv_cache_usage=[20.0, 95.5, 60.5], waiting_requests=[0, 4, 1], kv_cache_capacity=401856)

    assert stats.observations() == [
        cloudai.metrics.MetricObservation(cloudai.metrics.KV_CACHE_CAPACITY, 401856, {}),
        cloudai.metrics.MetricObservation(cloudai.metrics.KV_CACHE_USAGE, 58.666666666666664, {"statistic": "mean"}),
        cloudai.metrics.MetricObservation(cloudai.metrics.KV_CACHE_USAGE, 95.5, {"statistic": "max"}),
        cloudai.metrics.MetricObservation(cloudai.metrics.WAITING_REQUESTS, 4, {"statistic": "max"}),
    ]
    assert stats.observations(include_runtime=False) == stats.observations()[:1]


def test_vllm_metric_observations_include_serve_stats(slurm_system: SlurmSystem, vllm_tr: TestRun) -> None:
    test = cast(VllmTestDefinition, vllm_tr.test)
    bench_only = test.metric_observations(slurm_system, vllm_tr)
    (vllm_tr.output_path / VLLM_SERVE_LOG_FILE).write_text(SERVE_LOG, encoding="utf-8")

    observations = test.metric_observations(slurm_system, vllm_tr)

    assert observations[: len(bench_only)] == bench_only
    assert {obs.metric.key for obs in observations[len(bench_only) :]} == {
        "kv_cache_capacity",
        "kv_cache_usage",
        "waiting_requests",
    }


def test_vllm_metric_observations_skip_runtime_stats_with_semantic_eval(
    slurm_system: SlurmSystem, vllm_tr: TestRun
) -> None:
    test = cast(VllmTestDefinition, vllm_tr.test)
    test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs()
    (vllm_tr.output_path / VLLM_GSM8K_JSON_FILE).write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
    (vllm_tr.output_path / VLLM_SERVE_LOG_FILE).write_text(SERVE_LOG, encoding="utf-8")

    keys = [obs.metric.key for obs in test.metric_observations(slurm_system, vllm_tr)]

    assert "kv_cache_capacity" in keys
    assert "kv_cache_usage" not in keys
    assert "waiting_requests" not in keys


def test_vllm_metric_observations_ignore_serve_log_without_bench_results(
    slurm_system: SlurmSystem, vllm_tr: TestRun
) -> None:
    (vllm_tr.output_path / VLLM_BENCH_JSON_FILE).unlink()
    (vllm_tr.output_path / VLLM_SERVE_LOG_FILE).write_text(SERVE_LOG, encoding="utf-8")

    assert vllm_tr.test.metric_observations(slurm_system, vllm_tr) == []
