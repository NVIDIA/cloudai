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

import pytest

from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import (
    SGLangBenchReportGenerationStrategy,
    SglangCmdArgs,
    SglangTestDefinition,
)
from cloudai.workloads.sglang.sglang import SGLANG_BENCH_JSONL_FILE


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
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 30,
                "request_throughput": 2400.0,
                "max_concurrency": 16,
                "mean_ttft_ms": 120.0,
                "median_ttft_ms": 100.0,
                "p99_ttft_ms": 200.0,
                "mean_tpot_ms": 12.0,
                "median_tpot_ms": 10.0,
                "p99_tpot_ms": 20.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tr


def test_sglang_can_handle_directory(slurm_system: SlurmSystem, sglang_tr: TestRun) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    assert strategy.can_handle_directory() is True


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


def test_sglang_tps_per_gpu(slurm_system: SlurmSystem, sglang_tr: TestRun) -> None:
    strategy = SGLangBenchReportGenerationStrategy(slurm_system, sglang_tr)
    strategy.used_gpus_count = lambda: 4

    metric = strategy.get_metric("tps-per-gpu")

    assert metric == 600.0


def test_sglang_parses_num_prompts_from_input_lens(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = SglangTestDefinition(
        name="sglang_test_missing_num_prompts",
        description="SGLang benchmark",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev"),
    )
    tr = TestRun(name="sglang", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    (tr.output_path / SGLANG_BENCH_JSONL_FILE).write_text(
        json.dumps(
            {
                "completed": 30,
                "request_throughput": 2400.0,
                "max_concurrency": 16,
                "mean_ttft_ms": 120.0,
                "median_ttft_ms": 100.0,
                "p99_ttft_ms": 200.0,
                "mean_tpot_ms": 12.0,
                "median_tpot_ms": 10.0,
                "p99_tpot_ms": 20.0,
                "input_lens": [16] * 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    strategy = SGLangBenchReportGenerationStrategy(slurm_system, tr)
    assert strategy.can_handle_directory() is True
    assert strategy.get_metric("throughput") == 2400.0
