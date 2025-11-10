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

import numpy as np
import pytest

from cloudai import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nemo_launcher import NeMoLauncherReportGenerationStrategy


@pytest.fixture
def nemo_tr(tmp_path: Path) -> TestRun:
    tr = TestRun(name="nemo_launcher", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)

    log_content = (
        "[NeMo I 2025-03-04 02:58:29 perf_metrics_utils:56] train_step_timing in s: "
        "[11.85, 6.98, 5.36, 4.55, 4.12, 2.18, 2.18, 2.18, 2.18, 2.12, "
        "2.12, 2.12, 2.12, 2.12, 2.18, 2.18, 2.26, 2.45, 2.45, 2.39, "
        "2.46, 2.46, 2.28, 2.28, 2.28, 2.22, 2.14, 2.14, 2.14, 2.14, "
        "2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, "
        "2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.14, 2.13, 2.13]\n"
    )

    run_dir = tr.output_path / "run"
    run_dir.mkdir()
    (run_dir / "log-20250304_025354:run_438199.out").write_text(log_content)

    return tr


@pytest.fixture
def nemo_tr_large(tmp_path: Path) -> TestRun:
    tr = TestRun(name="nemo_launcher_large", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)

    step_timings = [float(i) for i in range(150)]
    log_content = f"[NeMo I 2025-03-04 02:58:29 perf_metrics_utils:56] train_step_timing in s: {step_timings}\n"

    run_dir = tr.output_path / "run"
    run_dir.mkdir()
    (run_dir / "log-20250304_025354:run_438199.out").write_text(log_content)

    return tr


def test_nemo_launcher_can_handle_directory(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr)
    assert strategy.can_handle_directory()


def test_nemo_launcher_extract_train_step_timings(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr)
    timings = strategy.extract_train_step_timings()

    assert timings, "No train step timings extracted."
    assert len(timings) == 50, "Expected 50 train step timing values."
    assert timings[:3] == [11.85, 6.98, 5.36], "First three timing values do not match expected."


def test_nemo_launcher_extract_train_step_timings_large(slurm_system: SlurmSystem, nemo_tr_large: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr_large)
    timings = strategy.extract_train_step_timings()

    assert timings, "No train step timings extracted."
    assert len(timings) == 20, "Expected last 20 train step timing values."
    assert timings == list(range(130, 150)), "Filtered timings do not match expected."


def test_nemo_launcher_generate_statistics_report(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr)
    timings = strategy.extract_train_step_timings()
    strategy.generate_statistics_report(timings)

    summary_file = nemo_tr.output_path / "train_step_timing_report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "avg": np.mean(timings),
        "median": np.median(timings),
        "min": np.min(timings),
        "max": np.max(timings),
    }

    for line in summary_content:
        key, value = line.lower().split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


def test_nemo_launcher_generate_bokeh_report(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr)
    timings = strategy.extract_train_step_timings()
    strategy.generate_bokeh_report(timings)

    report_file = nemo_tr.output_path / "cloudai_nemo_launcher_bokeh_report.html"
    assert report_file.is_file(), "Bokeh report was not generated."


def test_nemo_launcher_generate_report(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoLauncherReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "train_step_timing_report.txt"
    report_file = nemo_tr.output_path / "cloudai_nemo_launcher_bokeh_report.html"

    assert summary_file.is_file(), "Summary report was not generated."
    assert report_file.is_file(), "Bokeh report was not generated."
