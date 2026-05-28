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

from pathlib import Path

import pytest

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
from cloudai.workloads.ai_dynamo.ai_dynamo import parse_aiperf_accuracy
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
                    cmd="python3 -m dynamo.vllm --is-prefill-worker",
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
                    cmd="python3 -m dynamo.vllm --is-prefill-worker",
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
                    cmd="python3 -m dynamo.vllm --is-prefill-worker",
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
                    cmd="python3 -m dynamo.vllm --is-prefill-worker",
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


def test_ai_dynamo_get_metric_aiperf(slurm_system: SlurmSystem, ai_dynamo_aiperf_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_aiperf_tr)

    # aiperf fixture uses workloads="aiperf.sh" — bare names resolve to aiperf_report.csv.
    assert strategy.get_metric("Inter Token Latency (ms)") == 2.83
    assert strategy.get_metric("Output Token Throughput (tokens/sec)") == 595.68

    # Explicit prefix.
    assert strategy.get_metric("aiperf:Inter Token Latency (ms):avg") == 2.83
    assert strategy.get_metric("aiperf:Time to First Token (ms):avg") == 49.87
    assert strategy.get_metric("aiperf:Output Token Throughput (tokens/sec):avg") == 595.68
    assert strategy.get_metric("aiperf:Total Token Throughput (tokens/sec):avg") == 954.47


def test_ai_dynamo_get_metric_split_aiperf_accuracy(
    slurm_system: SlurmSystem, ai_dynamo_aiperf_with_split_accuracy_tr: TestRun
) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_aiperf_with_split_accuracy_tr)

    assert strategy.get_metric("accuracy") == 0.35
    assert strategy.get_metric("Inter Token Latency (ms)") == 2.83


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


def test_parse_aiperf_accuracy_from_split_accuracy_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "aiperf_accuracy_artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "accuracy_results.csv").write_text(get_aiperf_accuracy_csv_content(), encoding="utf-8")

    assert parse_aiperf_accuracy(tmp_path) == 0.35


def test_parse_aiperf_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    (tmp_path / "accuracy_results.csv").write_text("Task,Correct,Total,Accuracy\nOVERALL,n/a,100,n/a\n")

    assert parse_aiperf_accuracy(tmp_path) is None
