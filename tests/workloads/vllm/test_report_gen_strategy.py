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

from pathlib import Path

import pytest

from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.vllm import (
    VLLMBenchReport,
    VLLMBenchReportGenerationStrategy,
    VllmCmdArgs,
    VllmTestDefinition,
)
from cloudai.workloads.vllm.vllm import VLLM_BENCH_JSON_FILE

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


@pytest.mark.parametrize("ngpus", [1, 2, 4, 8])
def test_vllm_tps_per_gpu(slurm_system: SlurmSystem, vllm_tr: TestRun, ngpus: int) -> None:
    strategy = VLLMBenchReportGenerationStrategy(slurm_system, vllm_tr)
    strategy.used_gpus_count = lambda: ngpus

    metric = strategy.get_metric("tps-per-gpu")

    assert metric == BENCH_DATA.throughput / ngpus
