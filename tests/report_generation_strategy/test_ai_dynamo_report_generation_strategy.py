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
    GenAIPerf,
    LMBench,
    LMCache,
    LMCacheArgs,
    PrefillWorkerArgs,
)
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


@pytest.fixture
def ai_dynamo_tr(tmp_path: Path) -> TestRun:
    test = AIDynamoTestDefinition(
        name="ai_dynamo",
        description="desc",
        test_template_name="t",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="http://url",
            dynamo=AIDynamoArgs(prefill_worker=PrefillWorkerArgs()),
            genai_perf=GenAIPerf(),
            lmcache=LMCache(args=LMCacheArgs()),
            lmbench=LMBench(),
        ),
    )
    tr = TestRun(name="ai_dynamo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    csv_content = get_csv_content()
    # Create CSV file with the name expected by the new implementation
    (tr.output_path / "genai_perf_report.csv").write_text(csv_content)
    # Also create the file pattern expected by was_run_successful
    (tr.output_path / "profile_genai_perf.csv").write_text(csv_content)
    (tr.output_path / "profile_genai_perf.json").write_text("mock json content")

    return tr


@pytest.fixture
def csv_content() -> str:
    return get_csv_content()


def test_ai_dynamo_can_handle_directory(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)
    assert strategy.can_handle_directory() is True


def test_ai_dynamo_generate_report(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun, csv_content: str) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)
    # The new implementation does not generate a report file
    strategy.generate_report()
    # Just verify the method runs without error
    assert True


def test_ai_dynamo_get_metric_single_values(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    # Test that metrics from the first CSV section work
    assert strategy.get_metric("Output Sequence Length (tokens)") == 101.01
    assert strategy.get_metric("Input Sequence Length (tokens)") == 123.45


def test_ai_dynamo_get_metric_statistical_values(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    # Use exact metric names from CSV (with avg column, which is default)
    assert strategy.get_metric("Time To First Token (ms)") == 111.12
    assert strategy.get_metric("Time To Second Token (ms)") == 11.13
    assert strategy.get_metric("Request Latency (ms)") == 1111.14
    assert strategy.get_metric("Inter Token Latency (ms)") == 12.34


def test_ai_dynamo_get_metric_invalid(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    assert strategy.get_metric("invalid-metric") == METRIC_ERROR

    # Empty the CSV file to test error handling
    (ai_dynamo_tr.output_path / "genai_perf-report.csv").write_text("")
    assert strategy.get_metric("invalid-metric") == METRIC_ERROR


def test_was_run_successful(ai_dynamo_tr: TestRun) -> None:
    test_def = ai_dynamo_tr.test
    result = test_def.was_run_successful(ai_dynamo_tr)
    assert result.is_successful is True


def test_was_run_successful_no_results(ai_dynamo_tr: TestRun, tmp_path: Path) -> None:
    test_def = ai_dynamo_tr.test
    ai_dynamo_tr.output_path = tmp_path / "empty_output"
    ai_dynamo_tr.output_path.mkdir(parents=True, exist_ok=True)
    result = test_def.was_run_successful(ai_dynamo_tr)
    assert result.is_successful is False
