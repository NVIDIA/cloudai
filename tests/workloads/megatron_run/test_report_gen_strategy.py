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
import json
from pathlib import Path

import pytest
import toml
from click.testing import CliRunner

import cloudai.metrics
import cloudai.models.output
from cloudai import TestRun
from cloudai.cli import main
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


def test_generate_report_refreshes_megatron_observations(
    tmp_path: Path, megatron_run_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied results retain their execution metadata and acquire metrics without runtime access."""

    def unexpected_runtime_access(*args, **kwargs):
        pytest.fail("Report generation must only use disk artifacts")

    monkeypatch.setattr("subprocess.run", unexpected_runtime_access)
    result_dir = tmp_path / "copied-results"
    output_path = result_dir / "case" / "0"
    output_path.mkdir(parents=True)
    (output_path / "stdout.txt").write_text((megatron_run_tr.output_path / "stdout.txt").read_text())
    system_path = tmp_path / "system.toml"
    system_path.write_text(
        'name = "offline"\nscheduler = "slurm"\ninstall_path = "install"\n'
        'output_path = "results"\ndefault_partition = "main"\n[[partitions]]\nname = "main"\n'
    )
    scenario_path = tmp_path / "scenario.toml"
    config = megatron_run_tr.test.model_dump(mode="json", exclude_none=True)
    config.update(id="case", test_template_name="MegatronRun")
    scenario_path.write_text(toml.dumps({"name": "offline", "Tests": [config]}))
    experiment = cloudai.models.output.Experiment(
        id="original-id",
        name="offline",
        system_name="original-cluster",
        status="completed",
        path="/original/results",
        tests=[
            cloudai.models.output.Test(
                id="case",
                name="megatron_run",
                status="completed",
                path="/original/results/case",
                runs=[
                    cloudai.models.output.Run(
                        path="/original/results/case/0",
                        jobid="123",
                        status="completed",
                        duration=12.0,
                        iteration=0,
                        step=0,
                    )
                ],
            )
        ],
    )
    experiment_path = result_dir / "experiment.json"
    experiment_path.write_text(experiment.model_dump_json())
    result = CliRunner().invoke(
        main,
        [
            "--log-file",
            str(tmp_path / "debug.log"),
            "generate-report",
            "--system-config",
            str(system_path),
            "--test-scenario",
            str(scenario_path),
            "--result-dir",
            str(result_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    refreshed = json.loads(experiment_path.read_text())
    metrics = refreshed["tests"][0]["runs"][0].pop("metrics")
    assert refreshed["tests"][0].pop("metrics") == metrics
    original = experiment.model_dump(mode="json")
    original["tests"][0]["runs"][0].pop("metrics")
    original["tests"][0].pop("metrics")
    assert refreshed == original
    assert metrics == [
        {"name": "Iteration time", "value": pytest.approx(15629.1666667), "unit": "ms", "dimensions": []},
        {"name": "Throughput per GPU", "value": pytest.approx(495.0666667), "unit": "TFLOP/s/GPU", "dimensions": []},
    ]
    assert (output_path / "megatron_run_report.csv").is_file()
