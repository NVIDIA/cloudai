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
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nemo_run import NeMoBaseRunCmdArgs, NeMoRunReportGenerationStrategy, NeMoRunTestDefinition
from cloudai.workloads.nemo_run.report_generation_strategy import extract_timings


@pytest.fixture
def nemo_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="t",
            cmd_args=NeMoBaseRunCmdArgs(docker_image_url="docker://url", task="task", recipe_name="recipe"),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)
    tr.reports = {NeMoRunReportGenerationStrategy}

    stdout_content = (
        "Training epoch 0, iteration 17/99 | lr: 2.699e-06 | global_batch_size: 128 | global_step: 17 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2304\n"
        "Training epoch 0, iteration 18/99 | lr: 2.849e-06 | global_batch_size: 128 | global_step: 18 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2432\n"
        "Training epoch 0, iteration 19/99 | lr: 2.999e-06 | global_batch_size: 128 | global_step: 19 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2560\n"
        "Training epoch 0, iteration 20/99 | lr: 3.148e-06 | global_batch_size: 128 | global_step: 20 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 2688\n"
        "Training epoch 0, iteration 21/99 | lr: 3.298e-06 | global_batch_size: 128 | global_step: 21 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.87 | consumed_samples: 2816\n"
        "Training epoch 0, iteration 22/99 | lr: 3.448e-06 | global_batch_size: 128 | global_step: 22 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.87 | consumed_samples: 2944\n"
        "Training epoch 0, iteration 23/99 | lr: 3.598e-06 | global_batch_size: 128 | global_step: 23 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.63 | consumed_samples: 3072\n"
        "Training epoch 0, iteration 24/99 | lr: 3.748e-06 | global_batch_size: 128 | global_step: 24 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 13.04 | consumed_samples: 3200\n"
        "Training epoch 0, iteration 25/99 | lr: 3.898e-06 | global_batch_size: 128 | global_step: 25 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 3328\n"
        "Training epoch 0, iteration 26/99 | lr: 4.048e-06 | global_batch_size: 128 | global_step: 26 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3456\n"
        "Training epoch 0, iteration 27/99 | lr: 4.198e-06 | global_batch_size: 128 | global_step: 27 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3584\n"
        "Training epoch 0, iteration 28/99 | lr: 4.348e-06 | global_batch_size: 128 | global_step: 28 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3712\n"
    )

    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


@pytest.fixture
def nemo_tr_encoded(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="t",
            cmd_args=NeMoBaseRunCmdArgs(docker_image_url="docker://url", task="task", recipe_name="recipe"),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    stdout_content = (
        "┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃[1;35m [0m[1;35mArgument Name   [0m[1;35m [0m┃[1;35m [0m[1;35mResolved Value"
        "┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
        "│[2m [0m[2mdata            [0m[2m [0m│ [1;35mMockDataModule[0m[1m([0m[33mseq_length["
        "Training epoch 0, iteration 17/99 | lr: 2.699e-06 | global_batch_size: 128 | global_step: 17 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2304\n"
        "Training epoch 0, iteration 18/99 | lr: 2.849e-06 | global_batch_size: 128 | global_step: 18 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2432\n"
        "Training epoch 0, iteration 19/99 | lr: 2.999e-06 | global_batch_size: 128 | global_step: 19 | "
    )

    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


def test_nemo_can_handle_directory(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    assert strategy.can_handle_directory()


def test_nemo_generate_report(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.74,
        "Median": 12.65,
        "Min": 12.63,
        "Max": 13.04,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


def test_nemo_generate_report_encoded(slurm_system: SlurmSystem, nemo_tr_encoded: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_encoded)
    strategy.generate_report()

    summary_file = nemo_tr_encoded.output_path / "report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.74,
        "Median": 12.65,
        "Min": 12.63,
        "Max": 12.64,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


@pytest.mark.parametrize("metric", ["default", "step-time"])
def test_metrics(nemo_tr: TestRun, slurm_system: SlurmSystem, metric: str):
    nemo_tr.test.test_definition.agent_metric = metric
    value = nemo_tr.get_metric_value(slurm_system)
    assert value == 12.72090909090909


def test_extract_timings_valid_file(tmp_path: Path) -> None:
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text(
        "Training epoch 0, iteration 17/99 | train_step_timing in s: 12.64 | global_step: 17\n"
        "Training epoch 0, iteration 18/99 | train_step_timing in s: 12.65 | global_step: 18\n"
        "Training epoch 0, iteration 19/99 | train_step_timing in s: 12.66 | global_step: 19\n"
    )

    timings = extract_timings(stdout_file)
    assert timings == [12.65, 12.66], "Timings extraction failed for valid file."


def test_extract_timings_missing_file(tmp_path: Path) -> None:
    stdout_file = tmp_path / "missing_stdout.txt"

    timings = extract_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list for missing file."


def test_extract_timings_invalid_content(tmp_path: Path) -> None:
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("Invalid content without timing information\n")

    timings = extract_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list for invalid content."


def test_extract_timings_file_not_found(tmp_path: Path) -> None:
    stdout_file = tmp_path / "nonexistent_stdout.txt"

    timings = extract_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list when the file does not exist."


def test_generate_report_no_timings(slurm_system: SlurmSystem, nemo_tr: TestRun, tmp_path: Path) -> None:
    nemo_tr.output_path = tmp_path
    stdout_file = nemo_tr.output_path / "stdout.txt"
    stdout_file.write_text("No valid timing information\n")

    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert not summary_file.exists(), "Report should not be generated if no valid timings are found."


def test_generate_report_partial_timings(slurm_system: SlurmSystem, nemo_tr: TestRun, tmp_path: Path) -> None:
    nemo_tr.output_path = tmp_path
    stdout_file = nemo_tr.output_path / "stdout.txt"
    stdout_file.write_text(
        "Training epoch 0, iteration 17/99 | train_step_timing in s: 12.64 | global_step: 17\n"
        "Invalid line without timing\n"
        "Training epoch 0, iteration 18/99 | train_step_timing in s: 12.65 | global_step: 18\n"
    )

    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert summary_file.is_file(), "Report should be generated even with partial valid timings."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.645,
        "Median": 12.645,
        "Min": 12.64,
        "Max": 12.65,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."
