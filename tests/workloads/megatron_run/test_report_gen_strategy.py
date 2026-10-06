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

import csv
from pathlib import Path

import pytest

import cloudai.metrics
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
    report_path = megatron_run_tr.output_path / "megatron_run_report.csv"
    assert report_path.is_file()

    with report_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Should have 2 rows: iteration_time_ms and tflops_per_gpu
    assert len(rows) == 2

    expected_headers = {"metric_type", "avg", "median", "min", "max", "std"}
    assert set(rows[0].keys()) == expected_headers

    data = {row["metric_type"]: row for row in rows}

    # Verify iteration_time_ms stats
    assert "iteration_time_ms" in data
    iter_stats = data["iteration_time_ms"]
    expected_iter_avg = (15800.0 + 15639.0 + 15448.5) / 3
    assert abs(float(iter_stats["avg"]) - expected_iter_avg) < 0.1
    assert abs(float(iter_stats["median"]) - 15639.0) < 0.1
    assert abs(float(iter_stats["min"]) - 15448.5) < 0.1
    assert abs(float(iter_stats["max"]) - 15800.0) < 0.1

    # Verify tflops_per_gpu stats
    assert "tflops_per_gpu" in data
    tflops_stats = data["tflops_per_gpu"]
    expected_tflops_avg = (490.0 + 494.6 + 500.6) / 3
    assert abs(float(tflops_stats["avg"]) - expected_tflops_avg) < 0.1
    assert abs(float(tflops_stats["median"]) - 494.6) < 0.1
    assert abs(float(tflops_stats["min"]) - 490.0) < 0.1
    assert abs(float(tflops_stats["max"]) - 500.6) < 0.1


def test_megatron_run_get_metric_iteration_time(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Expected: avg of [15800.0, 15639.0, 15448.5]
    expected_avg = (15800.0 + 15639.0 + 15448.5) / 3
    metric = strategy.get_metric("iteration-time")
    assert metric is not METRIC_ERROR
    assert isinstance(metric, float)
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_default(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Default should return iteration-time
    expected_avg = (15800.0 + 15639.0 + 15448.5) / 3
    metric = strategy.get_metric("default")
    assert metric is not METRIC_ERROR
    assert isinstance(metric, float)
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_tflops(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    # Expected: avg of [490.0, 494.6, 500.6]
    expected_avg = (490.0 + 494.6 + 500.6) / 3
    metric = strategy.get_metric("tflops-per-gpu")
    assert metric is not METRIC_ERROR
    assert isinstance(metric, float)
    assert abs(metric - expected_avg) < 0.1


def test_megatron_run_get_metric_invalid(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    metric = strategy.get_metric("invalid-metric")
    assert metric is METRIC_ERROR


def test_megatron_run_get_metric_no_data(slurm_system: SlurmSystem, megatron_run_tr_no_data: TestRun) -> None:
    strategy = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr_no_data)
    metric = strategy.get_metric("iteration-time")
    assert metric is METRIC_ERROR


def test_megatron_run_metrics_class_var() -> None:
    assert MegatronRunReportGenerationStrategy.metrics == ["default", "iteration-time", "tflops-per-gpu"]


def test_metric_observations_use_last_ten_iterations(slurm_system: SlurmSystem, megatron_run_tr: TestRun) -> None:
    tr = megatron_run_tr
    (tr.output_path / "stdout.txt").write_text(
        "startup noise\n"
        + "\n".join(
            f"iteration {step} | elapsed time per iteration (ms): {step * 10}.0 | "
            f"throughput per GPU (TFLOP/s/GPU): {step}.0 |"
            for step in range(1, 13)
        )
    )
    observations = tr.test.metric_observations(slurm_system, tr)
    assert observations == [
        cloudai.metrics.MetricObservation(cloudai.metrics.ITERATION_TIME, 75.0, {}),
        cloudai.metrics.MetricObservation(cloudai.metrics.TFLOPS_PER_GPU, 7.5, {}),
    ]
    assert observations[0].metric.direction is cloudai.metrics.OptimizationDirection.MINIMIZE
    assert observations[1].metric.direction is cloudai.metrics.OptimizationDirection.MAXIMIZE
    report = MegatronRunReportGenerationStrategy(slurm_system, tr)
    assert report.get_metric("iteration-time") == observations[0].value
    assert report.get_metric("tflops-per-gpu") == observations[1].value
    report.generate_report()
    assert tr.test.metric_observations(slurm_system, tr) == observations


@pytest.mark.parametrize("stdout", [None, "validation loss at iteration 1.0\n", ""])
def test_metric_observations_without_iteration_data(
    slurm_system: SlurmSystem, megatron_run_tr: TestRun, stdout: str | None
) -> None:
    path = megatron_run_tr.output_path / "stdout.txt"
    if stdout is None:
        path.unlink()
    else:
        path.write_text(stdout)
    assert megatron_run_tr.test.metric_observations(slurm_system, megatron_run_tr) == []


def test_metrics_cache_uses_absolute_paths(
    tmp_path: Path, slurm_system: SlurmSystem, megatron_run_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reports and observations share a scan without confusing relative paths across directories."""
    open_calls = []
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        open_calls.append(path)
        return original_open(path, *args, **kwargs)

    for name, timing in (("first", 10), ("second", 20)):
        output = tmp_path / name / "results"
        output.mkdir(parents=True)
        (output / "stdout.txt").write_text(
            f"elapsed time per iteration (ms): {timing} | throughput per GPU (TFLOP/s/GPU): 100 |\n"
        )
    monkeypatch.setattr(Path, "open", record_open)
    megatron_run_tr.output_path = Path("results")
    report = MegatronRunReportGenerationStrategy(slurm_system, megatron_run_tr)
    for name, timing in (("first", 10), ("second", 20)):
        monkeypatch.chdir(tmp_path / name)
        assert report.can_handle_directory()
        assert report.get_metric("iteration-time") == timing
        assert megatron_run_tr.test.metric_observations(slurm_system, megatron_run_tr)[0].value == timing
    assert open_calls == [tmp_path / name / "results" / "stdout.txt" for name in ("first", "second")]
