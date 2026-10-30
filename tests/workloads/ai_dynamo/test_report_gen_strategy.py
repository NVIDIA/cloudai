# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

import cloudai.metrics
import cloudai.report_generator.inference_report as inference_report
import cloudai.util
from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoTestDefinition,
    AIPerf,
    AIPerfAccuracy,
    GenAIPerf,
    WorkerBaseArgs,
    WorkerConfig,
)
from cloudai.workloads.ai_dynamo.ai_dynamo import parse_aiperf_accuracy, parse_aiperf_request_count
from cloudai.workloads.ai_dynamo.report_generation_strategy import AIDynamoReportGenerationStrategy


def get_csv_content() -> str:
    return (
        "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\n"
        'Time To First Token (ms),111.12,222.22,"333.33","444.44","555.55","666.66",777.77,888.88,999.99\n'
        "Time To Second Token (ms),11.13,22.22,33.33,44.44,55.55,66.66,77.77,88.88,99.99\n"
        'Request Latency (ms),"1111.14","2222.22","3333.33","4444.44","5555.55","6666.66",'
        '"7777.77","8888.88","9999.99"\n'
        "Inter Token Latency (ms),12.34,23.45,34.56,45.67,56.78,67.89,78.90,89.01,90.12\n"
        "Output Sequence Length (tokens),101.01,202.02,303.03,404.04,505.05,606.06,707.07,808.08,909.09\n"
        "Input Sequence Length (tokens),123.45,234.56,345.67,456.78,567.89,678.90,789.01,890.12,901.23\n"
        "Output Token Throughput (tokens/sec),200\n"
        "Request Throughput (requests/sec),4\n"
        "Request Count (count),50\n"
    )


def get_aiperf_csv_content() -> str:
    return (
        "Metric,avg,min,max\n"
        "Inter Token Latency (ms),2.83,2.78,2.91\n"
        "Time to First Token (ms),49.87,17.15,99.91\n"
        "Output Sequence Length (tokens),498.06,410.00,501.00\n"
        "\n"
        "Metric,Value\n"
        "Output Token Throughput (tokens/sec),595.68\n"
        "Total Token Throughput (tokens/sec),954.47\n"
        "Request Count,50.00\n"
        "Request Throughput (requests/sec),1.25\n"
    )


def get_aiperf_accuracy_csv_content() -> str:
    return "Task,Correct,Total,Accuracy\nabstract_algebra,35,100,35.00%\nOVERALL,35,100,35.00%\n"


def get_aiperf_accuracy_cli() -> str:
    return "--model {model} --url {url} --artifact-dir {artifact_dir} --accuracy-benchmark mmlu"


@pytest.fixture
def ai_dynamo_tr(tmp_path: Path) -> TestRun:
    test = AIDynamoTestDefinition(
        name="ai_dynamo",
        description="desc",
        test_template_name="t",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="http://url",
            workloads="genai_perf.sh",
            dynamo=AIDynamoArgs(
                prefill_worker=WorkerConfig(
                    cmd="python3 -m dynamo.vllm",
                    worker_initialized_regex="VllmWorker.*has.been.initialized",
                    args=WorkerBaseArgs(),
                ),
            ),
            genai_perf=GenAIPerf(),
        ),
    )
    tr = TestRun(name="ai_dynamo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    csv_content = get_csv_content()
    (tr.output_path / "genai_perf_report.csv").write_text(csv_content)
    (tr.output_path / "aiperf_report.csv").write_text(get_aiperf_csv_content())
    (tr.output_path / "profile_genai_perf.csv").write_text(csv_content)
    (tr.output_path / "profile_genai_perf.json").write_text("mock json content")
    (tr.output_path / test.success_marker).touch()

    return tr


@pytest.fixture
def ai_dynamo_aiperf_tr(tmp_path: Path) -> TestRun:
    test = AIDynamoTestDefinition(
        name="ai_dynamo_aiperf",
        description="desc",
        test_template_name="t",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="http://url",
            workloads="aiperf.sh",
            dynamo=AIDynamoArgs(
                prefill_worker=WorkerConfig(
                    cmd="python3 -m dynamo.vllm",
                    worker_initialized_regex="VllmWorker.*has.been.initialized",
                    args=WorkerBaseArgs(),
                ),
            ),
            aiperf=AIPerf(),
        ),
    )
    tr = TestRun(name="ai_dynamo_aiperf", test=test, num_nodes=1, nodes=[], output_path=tmp_path)
    (tr.output_path / "aiperf_report.csv").write_text(get_aiperf_csv_content())
    (tr.output_path / test.success_marker).touch()
    return tr


@pytest.fixture
def ai_dynamo_aiperf_with_split_accuracy_tr(tmp_path: Path) -> TestRun:
    test = AIDynamoTestDefinition(
        name="ai_dynamo_aiperf_with_split_accuracy",
        description="desc",
        test_template_name="t",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="http://url",
            workloads="aiperf.sh",
            dynamo=AIDynamoArgs(
                prefill_worker=WorkerConfig(
                    cmd="python3 -m dynamo.vllm",
                    worker_initialized_regex="VllmWorker.*has.been.initialized",
                    args=WorkerBaseArgs(),
                ),
            ),
            aiperf=AIPerf(),
            aiperf_accuracy=AIPerfAccuracy.model_validate({"cli": get_aiperf_accuracy_cli()}),
        ),
    )
    tr = TestRun(name="ai_dynamo_aiperf_with_split_accuracy", test=test, num_nodes=1, nodes=[], output_path=tmp_path)
    (tr.output_path / "aiperf_report.csv").write_text(get_aiperf_csv_content())
    (tr.output_path / "accuracy_results.csv").write_text(get_aiperf_accuracy_csv_content())
    (tr.output_path / test.success_marker).touch()
    return tr


@pytest.fixture
def ai_dynamo_genai_perf_with_split_accuracy_tr(tmp_path: Path) -> TestRun:
    test = AIDynamoTestDefinition(
        name="ai_dynamo_genai_perf_with_split_accuracy",
        description="desc",
        test_template_name="t",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="http://url",
            workloads="genai_perf.sh",
            dynamo=AIDynamoArgs(
                prefill_worker=WorkerConfig(
                    cmd="python3 -m dynamo.vllm",
                    worker_initialized_regex="VllmWorker.*has.been.initialized",
                    args=WorkerBaseArgs(),
                ),
            ),
            genai_perf=GenAIPerf(),
            aiperf_accuracy=AIPerfAccuracy.model_validate({"cli": get_aiperf_accuracy_cli()}),
        ),
    )
    tr = TestRun(
        name="ai_dynamo_genai_perf_with_split_accuracy", test=test, num_nodes=1, nodes=[], output_path=tmp_path
    )
    (tr.output_path / "genai_perf_report.csv").write_text(get_csv_content())
    (tr.output_path / "accuracy_results.csv").write_text(get_aiperf_accuracy_csv_content())
    (tr.output_path / test.success_marker).touch()
    return tr


@pytest.fixture
def csv_content() -> str:
    return get_csv_content()


def test_ai_dynamo_can_handle_directory(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)
    assert strategy.can_handle_directory() is True


def test_ai_dynamo_generate_report(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun, csv_content: str) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)
    strategy.generate_report()
    assert True


def test_ai_dynamo_get_metric_genai_perf(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    # Default fixture uses workloads="genai_perf.sh" — bare names resolve to genai_perf_report.csv.
    assert strategy.get_metric("Inter Token Latency (ms)") == 12.34
    assert strategy.get_metric("Output Sequence Length (tokens)") == 101.01

    # Explicit prefix also works.
    assert strategy.get_metric("genai_perf:Time To First Token (ms):avg") == 111.12
    assert strategy.get_metric("genai_perf:Inter Token Latency (ms):p50") == 89.01
    observations = ai_dynamo_tr.test.metric_observations(slurm_system, ai_dynamo_tr)
    expected = [
        ("request_throughput", 4, {}),
        ("output_token_throughput", 200, {}),
        ("output_token_throughput_per_allocated_gpu", 200 / 8, {}),
        ("ttft", 111.12, {"statistic": "mean"}),
        ("ttft", 888.88, {"statistic": "median"}),
        ("ttft", 444.44, {"statistic": "p99"}),
        ("tpot", 12.34, {"statistic": "mean"}),
        ("tpot", 89.01, {"statistic": "median"}),
        ("tpot", 45.67, {"statistic": "p99"}),
    ]
    assert [(obs.metric.key, obs.value, dict(obs.dimensions)) for obs in observations] == expected
    ai_dynamo_tr.test.cmd_args.workloads = "genai_perf.sh,aiperf.sh"
    mixed = ai_dynamo_tr.test.metric_observations(slurm_system, ai_dynamo_tr)
    assert [(obs.metric.key, obs.value, dict(obs.dimensions)) for obs in mixed] == [
        (key, value, {"benchmark": "genai_perf", **dimensions}) for key, value, dimensions in expected
    ] + [
        ("request_throughput", 1.25, {"benchmark": "aiperf"}),
        ("output_token_throughput", 595.68, {"benchmark": "aiperf"}),
        ("output_token_throughput_per_allocated_gpu", 595.68 / 8, {"benchmark": "aiperf"}),
        ("ttft", 49.87, {"benchmark": "aiperf", "statistic": "mean"}),
        ("tpot", 2.83, {"benchmark": "aiperf", "statistic": "mean"}),
    ]


@pytest.mark.parametrize(
    "result", ["valid", "missing", "malformed", "nonfinite", "empty", "no-gpu-count", "missing-latency"]
)
def test_ai_dynamo_get_metric_aiperf(slurm_system: SlurmSystem, ai_dynamo_aiperf_tr: TestRun, result: str) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_aiperf_tr)

    # aiperf fixture uses workloads="aiperf.sh" — bare names resolve to aiperf_report.csv.
    assert strategy.get_metric("Inter Token Latency (ms)") == 2.83
    assert strategy.get_metric("Output Token Throughput (tokens/sec)") == 595.68

    # Explicit prefix.
    assert strategy.get_metric("aiperf:Inter Token Latency (ms):avg") == 2.83
    assert strategy.get_metric("aiperf:Time to First Token (ms):avg") == 49.87
    assert strategy.get_metric("aiperf:Output Token Throughput (tokens/sec):avg") == 595.68
    assert strategy.get_metric("aiperf:Total Token Throughput (tokens/sec):avg") == 954.47
    path = ai_dynamo_aiperf_tr.output_path / "aiperf_report.csv"
    if result == "missing":
        path.unlink()
    elif result == "malformed":
        path.write_text("Invalid CSV\n")
    elif result == "nonfinite":
        path.write_text(path.read_text().replace("595.68", "nan"))
    elif result == "empty":
        path.write_text(path.read_text().replace("Request Count,50.00", "Request Count,0"))
    elif result == "no-gpu-count":
        slurm_system.gpus_per_node = None
    elif result == "missing-latency":
        path.write_text(path.read_text().replace("Inter Token Latency (ms),2.83,2.78,2.91\n", ""))
        assert strategy.parse_results() is None
    expected = []
    if result not in {"missing", "malformed", "empty"}:
        expected.append(("request_throughput", 1.25, {}))
        if result != "nonfinite":
            expected.append(("output_token_throughput", 595.68, {}))
            if result != "no-gpu-count":
                expected.append(("output_token_throughput_per_allocated_gpu", 595.68 / 8, {}))
        expected.append(("ttft", 49.87, {"statistic": "mean"}))
        expected.extend([("tpot", 2.83, {"statistic": "mean"})] if result != "missing-latency" else [])
    observations = ai_dynamo_aiperf_tr.test.metric_observations(slurm_system, ai_dynamo_aiperf_tr)
    assert [(obs.metric.key, obs.value, dict(obs.dimensions)) for obs in observations] == expected


def test_ai_dynamo_get_metric_split_aiperf_accuracy(
    slurm_system: SlurmSystem, ai_dynamo_aiperf_with_split_accuracy_tr: TestRun
) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_aiperf_with_split_accuracy_tr)

    assert strategy.get_metric("accuracy") == 0.35
    assert strategy.get_metric("Inter Token Latency (ms)") == 2.83
    assert cloudai.metrics.MetricObservation(cloudai.metrics.ACCURACY, 0.35, {}) in (
        ai_dynamo_aiperf_with_split_accuracy_tr.test.metric_observations(
            slurm_system, ai_dynamo_aiperf_with_split_accuracy_tr
        )
    )


def test_ai_dynamo_accuracy_metric_requires_aiperf_accuracy_config(
    slurm_system: SlurmSystem, ai_dynamo_aiperf_tr: TestRun
) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_aiperf_tr)

    assert strategy.get_metric("accuracy") == METRIC_ERROR


def test_ai_dynamo_get_metric_invalid(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    assert strategy.get_metric("nonexistent-metric") == METRIC_ERROR

    (ai_dynamo_tr.output_path / "genai_perf_report.csv").write_text("")
    assert strategy.get_metric("Inter Token Latency (ms)") == METRIC_ERROR


def test_was_run_successful(ai_dynamo_tr: TestRun) -> None:
    test_def = ai_dynamo_tr.test
    result = test_def.was_run_successful(ai_dynamo_tr)
    assert result.is_successful is True


def test_was_run_successful_with_split_aiperf_accuracy(
    ai_dynamo_aiperf_with_split_accuracy_tr: TestRun,
) -> None:
    test_def = ai_dynamo_aiperf_with_split_accuracy_tr.test
    result = test_def.was_run_successful(ai_dynamo_aiperf_with_split_accuracy_tr)
    assert result.is_successful is True


def test_was_run_successful_rejects_aiperf_report_with_only_errors(ai_dynamo_aiperf_tr: TestRun) -> None:
    test_def = ai_dynamo_aiperf_tr.test
    (ai_dynamo_aiperf_tr.output_path / "aiperf_report.csv").write_text(
        "Metric,Value\nError Request Count,4.00\nTotal Error Input Sequence Length (tokens),1200.00\n"
    )

    result = test_def.was_run_successful(ai_dynamo_aiperf_tr)

    assert result.is_successful is False


def test_was_run_successful_with_genai_perf_and_split_aiperf_accuracy(
    ai_dynamo_genai_perf_with_split_accuracy_tr: TestRun,
) -> None:
    test_def = ai_dynamo_genai_perf_with_split_accuracy_tr.test
    result = test_def.was_run_successful(ai_dynamo_genai_perf_with_split_accuracy_tr)
    assert result.is_successful is True


def test_was_run_successful_requires_split_aiperf_accuracy(
    ai_dynamo_aiperf_with_split_accuracy_tr: TestRun,
) -> None:
    test_def = ai_dynamo_aiperf_with_split_accuracy_tr.test
    (ai_dynamo_aiperf_with_split_accuracy_tr.output_path / "accuracy_results.csv").unlink()
    result = test_def.was_run_successful(ai_dynamo_aiperf_with_split_accuracy_tr)
    assert result.is_successful is False


def test_was_run_successful_no_results(ai_dynamo_tr: TestRun, tmp_path: Path) -> None:
    test_def = ai_dynamo_tr.test
    ai_dynamo_tr.output_path = tmp_path / "empty_output"
    ai_dynamo_tr.output_path.mkdir(parents=True, exist_ok=True)
    result = test_def.was_run_successful(ai_dynamo_tr)
    assert result.is_successful is False


def test_parse_aiperf_accuracy_from_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "aiperf_artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "accuracy_results.csv").write_text(get_aiperf_accuracy_csv_content(), encoding="utf-8")

    assert parse_aiperf_accuracy(tmp_path) == 0.35


def test_parse_aiperf_request_count(tmp_path: Path) -> None:
    report = tmp_path / "aiperf_report.csv"
    report.write_text(get_aiperf_csv_content(), encoding="utf-8")

    request_count = parse_aiperf_request_count(report)

    assert request_count == 50
    assert isinstance(request_count, int)


def test_parse_aiperf_accuracy_from_split_accuracy_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "aiperf_accuracy_artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "accuracy_results.csv").write_text(get_aiperf_accuracy_csv_content(), encoding="utf-8")

    assert parse_aiperf_accuracy(tmp_path) == 0.35


def test_parse_aiperf_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    (tmp_path / "accuracy_results.csv").write_text("Task,Correct,Total,Accuracy\nOVERALL,n/a,100,n/a\n")

    assert parse_aiperf_accuracy(tmp_path) is None


INFERENCE_WORKER = {
    "num_nodes": 2,
    "cmd": "python -m dynamo.vllm",
    "worker_initialized_regex": "ready",
    "args": {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
        "dtype": "float16",
        "quantization": "fp8",
    },
}
INFERENCE_DEFINITION = {
    "name": "inference",
    "description": "Inference report",
    "test_template_name": "AIDynamo",
    "cmd_args": {
        "docker_image_url": "example/image:latest",
        "workloads": "aiperf.sh",
        "dynamo": {
            "model": "example/model",
            "backend": "vllm",
            "prefill_worker": INFERENCE_WORKER,
            "decode_worker": cloudai.util.deep_merge(INFERENCE_WORKER, {"args": {"pipeline_parallel_size": 2}}),
        },
        "aiperf": {
            "args": {
                "synthetic-input-tokens-mean": 128,
                "output-tokens-mean": 64,
                "request-count": 20,
                "concurrency": 4,
                "request-rate": 12.5,
            }
        },
        "aiperf_phases": [
            {"name": "low", "args": {"concurrency": 8}},
            {"name": "high", "args": {"concurrency": 32, "users": 16}},
        ],
    },
}
INFERENCE_ROLE = dict(
    model="example/model",
    num_nodes=2,
    num_instances=2,
    nodes_per_instance=1,
    instances_per_node=1,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    expert_parallel=True,
    precision="float16",
    quantization="fp8",
)


@pytest.mark.parametrize(
    "worker_overrides,backend,nodes,hardware,environment,role_overrides,decode_overrides,serving_gpus,shared_nodes",
    [
        pytest.param({}, "vllm", 4, 8, {}, {}, {}, 12, False, id="replicas"),
        pytest.param(
            {"multiple_workers_per_node": True},
            "vllm",
            4,
            8,
            {},
            {"num_instances": 8, "instances_per_node": 4},
            {"num_instances": 4, "instances_per_node": 2},
            32,
            False,
            id="packing",
        ),
        pytest.param({"multiple_workers_per_node": True}, "vllm", 2, 8, {}, {}, {}, 12, True, id="shared-nodes"),
        pytest.param(
            {"nodes_per_worker": 2},
            "vllm",
            4,
            8,
            {},
            {"num_instances": 1, "nodes_per_instance": 2, "instances_per_node": None},
            {"num_instances": 1, "nodes_per_instance": 2, "instances_per_node": None},
            6,
            False,
            id="multi-node-workers",
        ),
        pytest.param(
            {"args": {"data_parallel_size": 2}, "extra_args": "--enable-dp-attention"},
            "sglang",
            4,
            8,
            {},
            {"data_parallel_size": 2},
            {"data_parallel_size": 2},
            12,
            False,
            id="attention-dp",
        ),
        pytest.param(
            {"args": {"data_parallel_size": None, "pipeline_parallel_size": 1}},
            "sglang_dsr1",
            4,
            8,
            {},
            {
                "tensor_parallel_size": 16,
                "data_parallel_size": 16,
                "num_instances": 1,
                "nodes_per_instance": 2,
                "instances_per_node": None,
            },
            {
                "tensor_parallel_size": 16,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 16,
                "num_instances": 1,
                "nodes_per_instance": 2,
                "instances_per_node": None,
            },
            32,
            False,
            id="distributed-defaults",
        ),
        pytest.param(
            {"multiple_workers_per_node": True},
            "vllm",
            4,
            None,
            {},
            {"num_instances": None, "instances_per_node": None},
            {"num_instances": None, "instances_per_node": None},
            None,
            False,
            id="unknown-hardware",
        ),
        pytest.param(
            {"multiple_workers_per_node": True},
            "vllm",
            4,
            8,
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            {"num_instances": 4, "instances_per_node": 2},
            {},
            16,
            False,
            id="visible-gpus",
        ),
        pytest.param(
            {"extra_args": "--served-model-name tp --pipeline-parallel-size=1 --data-parallel-size 2"},
            "vllm",
            4,
            8,
            {},
            {"data_parallel_size": 2},
            {"pipeline_parallel_size": 1, "data_parallel_size": 2},
            16,
            False,
            id="argument-overrides",
        ),
        pytest.param(
            {"extra_args": "--tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} --data-parallel-size"},
            "vllm",
            4,
            8,
            {},
            {"tensor_parallel_size": None, "pipeline_parallel_size": None, "data_parallel_size": None},
            {"tensor_parallel_size": None, "pipeline_parallel_size": None, "data_parallel_size": None},
            None,
            False,
            id="unresolved-parallelism-overrides",
        ),
    ],
)
def test_inference_reports(
    slurm_system: SlurmSystem,
    tmp_path: Path,
    worker_overrides: dict,
    backend: str,
    nodes: int,
    hardware: int | None,
    environment: dict,
    role_overrides: dict,
    decode_overrides: dict,
    serving_gpus: int | None,
    shared_nodes: bool,
) -> None:
    definition = cloudai.util.deep_merge(
        INFERENCE_DEFINITION,
        {
            "cmd_args": {
                "dynamo": {"backend": backend, "prefill_worker": worker_overrides, "decode_worker": worker_overrides}
            },
            "extra_env_vars": environment,
        },
    )
    test = AIDynamoTestDefinition.model_validate(definition)
    tr = TestRun("inference", test, nodes, [], output_path=tmp_path)
    slurm_system.gpus_per_node = hardware
    expected = []
    for phase, concurrency, users, total in [("low", 8, None, 1500), ("high", 32, 16, 3000)]:
        artifact = tmp_path / "aiperf_artifacts" / phase
        artifact.mkdir(parents=True)
        (artifact / "profile_export_aiperf.json").write_text(
            json.dumps(
                {
                    "request_count": {"avg": 20},
                    "request_throughput": {"avg": 10},
                    "output_token_throughput": {"avg": 1000},
                    "total_token_throughput": {"avg": total},
                    "time_to_first_token": {"avg": 10, "p50": 11, "p99": 12},
                    "inter_token_latency": {"avg": 1, "p50": 2, "p99": 3},
                }
            )
        )
        (tmp_path / f"aiperf_{phase}_report.csv").write_text(get_aiperf_csv_content())
        metrics: list[tuple[cloudai.metrics.MetricDefinition, float, dict[str, str]]] = [
            (cloudai.metrics.REQUEST_THROUGHPUT, 10, {}),
            (cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT, 1000, {}),
        ]
        if hardware is not None:
            metrics.append((cloudai.metrics.OUTPUT_TOKEN_THROUGHPUT_PER_ALLOCATED_GPU, 1000 / (nodes * 8), {}))
        metrics.extend(
            (metric, value, {"statistic": statistic})
            for metric, values in ((cloudai.metrics.TTFT, (10, 11, 12)), (cloudai.metrics.TPOT, (1, 2, 3)))
            for statistic, value in zip(("mean", "median", "p99"), values, strict=True)
        )
        metrics.extend(
            [
                (cloudai.metrics.TOTAL_TOKEN_THROUGHPUT, total, {}),
                (cloudai.metrics.TOTAL_TOKEN_THROUGHPUT_PER_USER, total / (users or concurrency), {}),
            ]
        )
        if serving_gpus is not None:
            metrics.append((cloudai.metrics.TOTAL_TOKEN_THROUGHPUT_PER_SERVING_GPU, total / serving_gpus, {}))
        expected.append(
            inference_report.InferenceReport(
                config=inference_report.InferenceConfig(
                    model="example/model",
                    num_nodes=nodes,
                    backend=backend,
                    benchmark="aiperf",
                    phase=phase,
                    container_image="example/image:latest",
                    gpus_per_node=hardware,
                    serving_gpu_count=serving_gpus,
                    input_tokens=128,
                    output_tokens=64,
                    concurrency=concurrency,
                    users=users,
                    request_count=20,
                    request_rate=12.5,
                    prefill=inference_report.InferenceRoleConfig(**{**INFERENCE_ROLE, **role_overrides}),
                    decode=inference_report.InferenceRoleConfig(
                        **{**INFERENCE_ROLE, "pipeline_parallel_size": 2, **decode_overrides}
                    ),
                    shared_nodes=shared_nodes,
                ),
                observations=[
                    cloudai.metrics.MetricObservation(metric, value, {"phase": phase, **dimensions})
                    for metric, value, dimensions in metrics
                ],
            )
        )
    assert test.inference_reports(slurm_system, tr) == expected
    assert test.metric_observations(slurm_system, tr) == [obs for report in expected for obs in report.observations]
