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
from cloudai.workloads.megatron_run import (
    MegatronRunCmdArgs,
    MegatronRunReportGenerationStrategy,
    MegatronRunTestDefinition,
)


@pytest.fixture
def megatron_run_tr(tmp_path: Path) -> TestRun:
    test = MegatronRunTestDefinition(
        name="megatron_run",
        description="desc",
        test_template_name="t",
        cmd_args=MegatronRunCmdArgs(docker_image_url="http://url", run_script=Path(__file__)),
    )
    tr = TestRun(name="megatron_run_test", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    stdout_content = (
        "[2026-01-16 07:32:24] iteration        5/     100 | consumed samples:        10240 | "
        "elapsed time per iteration (ms): 15800.0 | throughput per GPU (TFLOP/s/GPU): 490.0 | "
        "learning rate: 4.134000E-07 | global batch size:  2048 | lm loss: 1.344240E+01 | "
        "seq_load_balancing_loss: 1.000203E+00 | loss scale: 1.0 | grad norm: 2.870 | "
        "num zeros: 1174412544.0 | params norm: 8660.607 | "
        "number of skipped iterations:   0 | number of nan iterations:   0 |\n"
        "[2026-01-16 07:32:39] iteration        6/     100 | consumed samples:        12288 | "
        "elapsed time per iteration (ms): 15639.0 | throughput per GPU (TFLOP/s/GPU): 494.6 | "
        "learning rate: 4.180800E-07 | global batch size:  2048 | lm loss: 1.342407E+01 | "
        "seq_load_balancing_loss: 1.000202E+00 | loss scale: 1.0 | grad norm: 2.867 | "
        "num zeros: 1174412672.0 | params norm: 8660.606 | "
        "number of skipped iterations:   0 | number of nan iterations:   0 |\n"
        "[2026-01-16 07:32:54] iteration        7/     100 | consumed samples:        14336 | "
        "elapsed time per iteration (ms): 15448.5 | throughput per GPU (TFLOP/s/GPU): 500.6 | "
        "learning rate: 4.227600E-07 | global batch size:  2048 | lm loss: 1.340574E+01 | "
        "seq_load_balancing_loss: 1.000201E+00 | loss scale: 1.0 | grad norm: 2.864 | "
        "num zeros: 1174412800.0 | params norm: 8660.605 | "
        "number of skipped iterations:   0 | number of nan iterations:   0 |\n"
    )
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


@pytest.fixture
def megatron_run_tr_no_data(tmp_path: Path) -> TestRun:
    test = MegatronRunTestDefinition(
        name="megatron_run",
        description="desc",
        test_template_name="t",
        cmd_args=MegatronRunCmdArgs(docker_image_url="http://url", run_script=Path(__file__)),
    )
    tr = TestRun(name="megatron_run_test", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    stdout_content = """
Some random log output without iteration metrics
Starting training...
"""
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


def test_megatron_run_can_handle_directory(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    assert strategy.can_handle_directory()


def test_megatron_run_cannot_handle_directory_without_iteration_data(
    slurm_system: SlurmSystem, megatron_run_tr_no_data: TestRun
) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_data)
    assert not strategy.can_handle_directory()


def test_megatron_run_extract_and_generate_report(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    strategy.generate_report()
    report_path = megatron_run_tr.output_path / "megatron_run_report.txt"
    assert report_path.is_file()
    content = report_path.read_text()
    assert "Iteration Time (ms)" in content
    assert "TFLOP/s per GPU" in content
    assert "avg:" in content
    assert "median:" in content
    assert "min:" in content
    assert "max:" in content
    assert "std:" in content


def test_megatron_run_get_metric_iteration_time(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Expected: avg of [15800.0, 15639.0, 15448.5]
    expected_avg = (15800.0 + 15639.0 + 15448.5) / 3
    metric = strategy.get_metric("iteration-time")
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_default(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Default should return iteration-time
    expected_avg = (15800.0 + 15639.0 + 15448.5) / 3
    metric = strategy.get_metric("default")
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_tflops(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Expected: avg of [490.0, 494.6, 500.6]
    expected_avg = (490.0 + 494.6 + 500.6) / 3
    metric = strategy.get_metric("tflops-per-gpu")
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_invalid(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    metric = strategy.get_metric("invalid-metric")
    assert metric == METRIC_ERROR


def test_megatron_run_get_metric_no_data(slurm_system: SlurmSystem, megatron_run_tr_no_data: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_data)
    metric = strategy.get_metric("iteration-time")
    assert metric == METRIC_ERROR


def test_megatron_run_metrics_class_var() -> None:
    assert MegatronRunReportGenerationStrategy.metrics == ["default", "iteration-time", "tflops-per-gpu"]
