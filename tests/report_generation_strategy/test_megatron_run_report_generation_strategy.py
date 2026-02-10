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
from unittest.mock import Mock

import pytest

from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.megatron_run import MegatronRunReportGenerationStrategy


@pytest.fixture
def megatron_run_tr(tmp_path: Path) -> TestRun:
    """Create a TestRun with sample Megatron-LM training logs."""
    tr = TestRun(name="megatron_run", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    log_content = (
        " [2026-02-06 20:55:02.918469] iteration      292/95367431 | consumed samples:         9344 | "
        "elapsed time per iteration (ms): 3075.7 | throughput per GPU (TFLOP/s/GPU): 478.0 | "
        "energy per GPU (J/iter/GPU): 1992.5 | power per GPU (W/GPU): 647.8 | learning rate: 9.568256E-08 | "
        "global batch size:    32 | lm loss: 2.401035E-02 | loss scale: 1.0 | grad norm: 1.797 |\n"
        " [2026-02-06 20:55:05.956222] iteration      293/95367431 | consumed samples:         9376 | "
        "elapsed time per iteration (ms): 3037.2 | throughput per GPU (TFLOP/s/GPU): 484.0 | "
        "energy per GPU (J/iter/GPU): 1982.6 | power per GPU (W/GPU): 652.8 | learning rate: 9.601024E-08 | "
        "global batch size:    32 | lm loss: 2.386082E-02 | loss scale: 1.0 | grad norm: 1.797 |\n"
        " [2026-02-06 20:55:08.991445] iteration      294/95367431 | consumed samples:         9408 | "
        "elapsed time per iteration (ms): 3035.2 | throughput per GPU (TFLOP/s/GPU): 484.3 | "
        "energy per GPU (J/iter/GPU): 1980.1 | power per GPU (W/GPU): 652.5 | learning rate: 9.633792E-08 | "
        "global batch size:    32 | lm loss: 2.378540E-02 | loss scale: 1.0 | grad norm: 1.796 |\n"
    )
    (tr.output_path / "stdout.txt").write_text(log_content)
    return tr


@pytest.fixture
def megatron_run_tr_no_metrics(tmp_path: Path) -> TestRun:
    """Create a TestRun with log file but no iteration metrics."""
    tr = TestRun(name="megatron_run", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    log_content = "Some log content without iteration metrics\nAnother line\n"
    (tr.output_path / "stdout.txt").write_text(log_content)
    return tr


@pytest.fixture
def megatron_run_tr_empty(tmp_path: Path) -> TestRun:
    """Create a TestRun with no log file."""
    tr = TestRun(name="megatron_run", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    return tr


def test_can_handle_directory(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test that strategy can handle directory with valid Megatron logs."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    assert strategy.can_handle_directory()


def test_cannot_handle_directory_no_file(slurm_system: SlurmSystem, megatron_run_tr_empty: TestRun) -> None:
    """Test that strategy cannot handle directory without log file."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_empty)
    assert not strategy.can_handle_directory()


def test_cannot_handle_directory_no_metrics(slurm_system: SlurmSystem, megatron_run_tr_no_metrics: TestRun) -> None:
    """Test that strategy cannot handle directory without iteration metrics."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_metrics)
    assert not strategy.can_handle_directory()


def test_generate_report(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test report generation with valid logs."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    strategy.generate_report()

    report_path = megatron_run_tr.output_path / "report.txt"
    assert report_path.is_file(), "Report file should be created."

    content = report_path.read_text()
    assert "Step Time (s)" in content
    assert "TFLOP/s per GPU" in content
    assert "avg:" in content
    assert "median:" in content
    assert "min:" in content
    assert "max:" in content
    assert "std:" in content


def test_generate_report_statistics(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test that report contains correct statistics."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    strategy.generate_report()

    report_path = megatron_run_tr.output_path / "report.txt"
    content = report_path.read_text()

    # Expected values based on the log content:
    # Step times (ms -> s): 3075.7 -> 3.0757, 3037.2 -> 3.0372, 3035.2 -> 3.0352
    # avg step time: (3.0757 + 3.0372 + 3.0352) / 3 = 3.0494
    # TFLOPs: 478.0, 484.0, 484.3
    # avg tflops: (478.0 + 484.0 + 484.3) / 3 = 482.1

    # Check step time avg is approximately correct (converting ms to s)
    assert "3.04" in content or "3.05" in content, "Average step time should be around 3.04-3.05 seconds"

    # Check tflops avg is approximately correct
    assert "482" in content, "Average TFLOP/s should be around 482"


def test_generate_report_no_metrics(slurm_system: SlurmSystem, megatron_run_tr_no_metrics: TestRun) -> None:
    """Test report generation when no metrics are found."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_metrics)
    strategy.generate_report()

    report_path = megatron_run_tr_no_metrics.output_path / "report.txt"
    assert report_path.is_file(), "Report file should be created even with no metrics."

    content = report_path.read_text()
    assert "No iteration metrics found" in content


def test_generate_report_no_file(slurm_system: SlurmSystem, megatron_run_tr_empty: TestRun) -> None:
    """Test report generation when log file is missing."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_empty)
    strategy.generate_report()

    report_path = megatron_run_tr_empty.output_path / "report.txt"
    assert not report_path.exists(), "Report file should not be created when log file is missing."


def test_get_metric_step_time(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test getting step-time metric."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    value = strategy.get_metric("step-time")

    # Expected: avg of 3.0757, 3.0372, 3.0352 = 3.0494 seconds
    assert value != METRIC_ERROR
    assert pytest.approx(value, rel=0.01) == 3.0494


def test_get_metric_default(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test that default metric returns step-time."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    value = strategy.get_metric("default")

    # Default should be the same as step-time
    step_time_value = strategy.get_metric("step-time")
    assert value == step_time_value


def test_get_metric_tflops(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test getting tflops-per-gpu metric."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    value = strategy.get_metric("tflops-per-gpu")

    # Expected: avg of 478.0, 484.0, 484.3 = 482.1
    assert value != METRIC_ERROR
    assert pytest.approx(value, rel=0.01) == 482.1


def test_get_metric_invalid(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test that invalid metric returns METRIC_ERROR."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    value = strategy.get_metric("invalid-metric")
    assert value == METRIC_ERROR


def test_get_metric_no_file(slurm_system: SlurmSystem, megatron_run_tr_empty: TestRun) -> None:
    """Test that missing log file returns METRIC_ERROR."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_empty)
    value = strategy.get_metric("step-time")
    assert value == METRIC_ERROR


def test_get_metric_no_metrics(slurm_system: SlurmSystem, megatron_run_tr_no_metrics: TestRun) -> None:
    """Test that log without metrics returns METRIC_ERROR."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_metrics)
    value = strategy.get_metric("step-time")
    assert value == METRIC_ERROR


def test_results_file_property(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    """Test that results_file property returns correct path."""
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    assert strategy.results_file == megatron_run_tr.output_path / "stdout.txt"


def test_metrics_class_variable() -> None:
    """Test that metrics class variable is correctly defined."""
    assert MegatronRunReportGenerationStrategy.metrics == ["default", "step-time", "tflops-per-gpu"]


def test_extract_with_partial_valid_lines(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    """Test extraction with some valid and some invalid log lines."""
    tr = TestRun(name="megatron_run", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    log_content = (
        "Some random log line\n"
        " [2026-02-06 20:55:02.918469] iteration      292/100 | "
        "elapsed time per iteration (ms): 3000.0 | throughput per GPU (TFLOP/s/GPU): 480.0 |\n"
        "Another random line without metrics\n"
        " [2026-02-06 20:55:05.956222] iteration      293/100 | "
        "elapsed time per iteration (ms): 3100.0 | throughput per GPU (TFLOP/s/GPU): 490.0 |\n"
        "Final random line\n"
    )
    (tr.output_path / "stdout.txt").write_text(log_content)

    strategy = MegatronRunReportGenerationStrategy(slurm_system, tr)

    # Should extract 2 valid samples
    step_time = strategy.get_metric("step-time")
    assert step_time != METRIC_ERROR
    # avg of 3.0 and 3.1 seconds = 3.05
    assert pytest.approx(step_time, rel=0.01) == 3.05

    tflops = strategy.get_metric("tflops-per-gpu")
    assert tflops != METRIC_ERROR
    # avg of 480 and 490 = 485
    assert pytest.approx(tflops, rel=0.01) == 485.0
