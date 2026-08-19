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
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.megatron_bridge import MegatronBridgeReportGenerationStrategy
from cloudai.workloads.megatron_bridge.metrics import extract_mbridge_metrics


@pytest.fixture
def mb_tr(tmp_path: Path) -> TestRun:
    tr = TestRun(name="megatron_bridge", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    log_content = "\n".join(
        [
            "ain_fp8_mx/0 Step Time : 9.09s GPU utilization: 663.5MODEL_TFLOP/s/GPU",
            "",
            "ain_fp8_mx/0  [2025-12-22 15:18:33] iteration       50/      50 | consumed samples:        25600 | ",
            "elapsed time per iteration (ms): 9089.0 | learning rate: 3.000000E-05 | global batch size:   512 | ",
            "lm loss: 8.114214E+00 | load_balancing_loss: 1.000000E+00 | loss scale: 1.0 | grad norm: 0.042 | ",
            "number of skipped iterations:   0 | number of nan iterations:   0 |",
            "",
        ]
    )
    (tr.output_path / "cloudai_megatron_bridge_launcher.log").write_text(log_content)
    return tr


def test_megatron_bridge_can_handle_directory(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)
    assert strategy.can_handle_directory()


def test_megatron_bridge_extract_and_generate_report(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)
    strategy.generate_report()
    report_path = mb_tr.output_path / "report.txt"
    assert report_path.is_file()
    content = report_path.read_text()
    assert "Step Time" in content
    assert "TFLOP/s per GPU" in content


def test_megatron_bridge_reads_nested_slurm_log(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "cloudai_megatron_bridge_launcher.log").write_text("- Job id: 123\n")
    slurm_log = mb_tr.output_path / "experiments" / "experiment" / "run" / "job" / "log-test_123_0.out"
    slurm_log.parent.mkdir(parents=True)
    slurm_log.write_text("model/0 Step Time : 7.5s GPU utilization: 700.0 TFLOP/s/GPU iteration 1/10\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 7.5
    assert strategy.results_file == slurm_log


def test_megatron_bridge_selects_nested_log_by_submitted_job_id(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "cloudai_megatron_bridge_launcher.log").write_text("- Job id: 123\n")
    logs_dir = mb_tr.output_path / "experiments" / "experiment" / "run" / "job"
    logs_dir.mkdir(parents=True)
    (logs_dir / "log-test_999_0.out").write_text("Step Time : 99s iteration 1/10\n")
    expected_log = logs_dir / "log-test_123_0.out"
    expected_log.write_text("Step Time : 7s iteration 1/10\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 7.0
    assert strategy.results_file == expected_log


def test_extract_mbridge_metrics_accepts_independent_lines() -> None:
    step_times, gpu_tflops = extract_mbridge_metrics(
        "model/0 Step Time : 8.25s\nmodel/0 GPU utilization: 612.5 MODEL_TFLOP/s/GPU\nmodel/0 iteration 3/10\n"
    )

    assert step_times == [8.25]
    assert gpu_tflops == [612.5]


def test_extract_mbridge_metrics_falls_back_to_iteration_time() -> None:
    step_times, gpu_tflops = extract_mbridge_metrics(
        "iteration 10/10 | elapsed time per iteration (ms): 9125.0 |\nGPU utilization: 601.0\n"
    )

    assert step_times == [9.125]
    assert gpu_tflops == [601.0]


def test_extract_mbridge_metrics_associates_separate_iteration_time_line() -> None:
    step_times, _ = extract_mbridge_metrics(
        "iteration 1/2\nelapsed time per iteration (ms): 4000.0\n"
        "iteration 2/2\nelapsed time per iteration (ms): 3000.0\n"
    )

    assert step_times == [4.0, 3.0]


def test_extract_mbridge_metrics_associates_values_after_iteration() -> None:
    step_times, gpu_tflops = extract_mbridge_metrics(
        "iteration 1/2\nStep Time : 4.0s\nGPU utilization: 500.0\n"
        "iteration 2/2\nStep Time : 3.0s\nGPU utilization: 600.0\n"
    )

    assert step_times == [4.0, 3.0]
    assert gpu_tflops == [500.0, 600.0]


def test_extract_mbridge_metrics_keeps_last_ten_iterations() -> None:
    log_data = "".join(
        f"Step Time : {iteration}.0s GPU utilization: {iteration * 10}.0 iteration {iteration}/12\n"
        for iteration in range(1, 13)
    )

    step_times, gpu_tflops = extract_mbridge_metrics(log_data)

    assert step_times == [float(iteration) for iteration in range(3, 13)]
    assert gpu_tflops == [float(iteration * 10) for iteration in range(3, 13)]


def test_extract_mbridge_metrics_requires_iterations_like_mbridge() -> None:
    step_times, gpu_tflops = extract_mbridge_metrics("Step Time : 8.25s GPU utilization: 612.5\n")

    assert step_times == []
    assert gpu_tflops == []


def test_megatron_bridge_uses_registered_job_logs(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "slurm-job.toml").write_text('job_id = "123"\n')
    registered_dir = mb_tr.output_path / "registered"
    registered_dir.mkdir()
    registered_log = registered_dir / "log-job_123_0.out"
    registered_log.write_text("Step Time : 6s iteration 1/1\n")
    (mb_tr.output_path / ".slurm_jobs").write_text(f"123 = {registered_dir}/log*,{registered_dir},LocalTunnel,{{}}\n")
    unrelated_log = mb_tr.output_path / "experiments" / "newer" / "log-job_999_0.out"
    unrelated_log.parent.mkdir(parents=True)
    unrelated_log.write_text("Step Time : 99s iteration 1/1\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 6.0
    assert strategy.results_file == registered_log


def test_megatron_bridge_prefers_allranks_logs(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "slurm-job.toml").write_text("job_id = 123\n")
    logs_dir = mb_tr.output_path / "registered"
    logs_dir.mkdir()
    (logs_dir / "log-rank0_123.out").write_text("Step Time : 99s iteration 1/1\n")
    allranks_log = logs_dir / "log-allranks_123.out"
    allranks_log.write_text("Step Time : 5s iteration 1/1\n")
    (mb_tr.output_path / ".slurm_jobs").write_text(f"123 = {logs_dir}/log*,{logs_dir},LocalTunnel,{{}}\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 5.0
    assert strategy.results_file == allranks_log


def test_megatron_bridge_merges_restart_logs_by_iteration(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "slurm-job.toml").write_text("job_id = 123\n")
    logs_dir = mb_tr.output_path / "registered"
    logs_dir.mkdir()
    (logs_dir / "log-allranks_123_0.out").write_text("Step Time : 8s iteration 1/2\n")
    latest_log = logs_dir / "log-allranks_123_1.out"
    latest_log.write_text("Step Time : 7s iteration 1/2\nStep Time : 6s iteration 2/2\n")
    (mb_tr.output_path / ".slurm_jobs").write_text(f"123 = {logs_dir}/log*,{logs_dir},LocalTunnel,{{}}\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 6.5


def test_megatron_bridge_tolerates_invalid_utf8(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    (mb_tr.output_path / "slurm-job.toml").write_text("job_id = 123\n")
    logs_dir = mb_tr.output_path / "registered"
    logs_dir.mkdir()
    log_file = logs_dir / "log-job_123_0.out"
    log_file.write_bytes(b"NCCL banner: \xff\nStep Time : 5s GPU utilization: 600 iteration 1/1\n")
    (mb_tr.output_path / ".slurm_jobs").write_text(f"123 = {log_file},{logs_dir},LocalTunnel,{{}}\n")

    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)

    assert strategy.get_metric("step-time") == 5.0
    assert strategy.get_metric("tflops-per-gpu") == 600.0
