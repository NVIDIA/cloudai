# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import Mock

import pytest

from cloudai import Test, TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoTestDefinition,
    DecodeWorkerArgs,
    GenAIPerfArgs,
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
        "Input Sequence Length (tokens),123.45,234.56,345.67,456.78,567.89,678.90,789.01,890.12,901.23\n\n"
        "Metric,Value\n"
        "Output Token Throughput (tokens/sec),24\n"
        "Request Throughput (per sec),1.23\n"
        "Request Count (count),40.00\n"
    )


@pytest.fixture
def ai_dynamo_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=AIDynamoTestDefinition(
            name="ai_dynamo",
            description="desc",
            test_template_name="t",
            cmd_args=AIDynamoCmdArgs(
                docker_image_url="http://url",
                dynamo=AIDynamoArgs(
                    prefill_worker=PrefillWorkerArgs(
                        **{
                            "num-nodes": 1,
                            "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                        }
                    ),
                    decode_worker=DecodeWorkerArgs(
                        **{
                            "num-nodes": 1,
                            "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                        }
                    ),
                ),
                genai_perf=GenAIPerfArgs(
                    **{
                        "streaming": False,
                        "extra-inputs": "mock_extra_inputs",
                        "input-file": "mock_input_file",
                        "output-tokens-mean": 100,
                        "random-seed": 123,
                        "request-count": 100,
                        "synthetic-input-tokens-mean": 100,
                        "warmup-request-count": 10,
                    }
                ),
            ),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="ai_dynamo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    csv_content = get_csv_content()
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
    strategy.generate_report()

    report_file = ai_dynamo_tr.output_path / "report.csv"
    assert report_file.is_file(), "Report CSV was not generated."

    report_content = report_file.read_text()
    expected_content = csv_content + "Overall Output Tokens per Second per GPU,1.0\n"
    assert report_content == expected_content, "Report content does not match expected."


def test_ai_dynamo_get_metric_single_values(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    assert strategy.get_metric("output-token-throughput") == 24.0
    assert strategy.get_metric("request-throughput") == 1.23
    assert strategy.get_metric("default") == 24.0


def test_ai_dynamo_get_metric_statistical_values(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    assert strategy.get_metric("time-to-first-token") == 111.12
    assert strategy.get_metric("time-to-second-token") == 11.13
    assert strategy.get_metric("request-latency") == 1111.14
    assert strategy.get_metric("inter-token-latency") == 12.34


def test_ai_dynamo_get_metric_invalid(slurm_system: SlurmSystem, ai_dynamo_tr: TestRun) -> None:
    strategy = AIDynamoReportGenerationStrategy(slurm_system, ai_dynamo_tr)

    assert strategy.get_metric("invalid-metric") == METRIC_ERROR

    (ai_dynamo_tr.output_path / "profile_genai_perf.csv").write_text("")
    assert strategy.get_metric("default") == METRIC_ERROR
