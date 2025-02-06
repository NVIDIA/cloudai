# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.schema.test_template.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy


@pytest.fixture
def nemo_test_environment(tmp_path: Path) -> Path:
    test_dir = tmp_path / "test_nemo_env"
    test_dir.mkdir()

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

    (test_dir / "stdout.txt").write_text(stdout_content)

    return test_dir


def test_nemo_can_handle_directory(nemo_test_environment: Path) -> None:
    strategy = NeMoRunReportGenerationStrategy()
    assert strategy.can_handle_directory(nemo_test_environment)


def test_nemo_generate_report(nemo_test_environment: Path) -> None:
    strategy = NeMoRunReportGenerationStrategy()
    strategy.generate_report("nemo_test", nemo_test_environment)

    summary_file = nemo_test_environment / "report.txt"
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
