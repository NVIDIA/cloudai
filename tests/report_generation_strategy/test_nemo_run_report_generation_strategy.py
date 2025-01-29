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
        "[NeMo I 2024-11-15 10:22:04 utils:259] Setting up optimizer with config "
        "OptimizerConfig(optimizer='adam', lr=0.0003)\n"
        "Training epoch 0, iteration 0/4 | lr: 1.499e-07 | consumed_samples: 512 | "
        "global_batch_size: 512 | global_step: 0 | reduced_train_loss: 11.03 | "
        "train_step_timing in s: 61.94\n"
        "Training epoch 0, iteration 1/4 | lr: 2.999e-07 | consumed_samples: 1024 | "
        "global_batch_size: 512 | global_step: 1 | reduced_train_loss: 11.03 | "
        "train_step_timing in s: 53.67\n"
        "Training epoch 0, iteration 2/4 | lr: 4.498e-07 | consumed_samples: 1536 | "
        "global_batch_size: 512 | global_step: 2 | reduced_train_loss: 11.03 | "
        "train_step_timing in s: 52.45\n"
        "Training epoch 0, iteration 3/4 | lr: 5.997e-07 | consumed_samples: 2048 | "
        "global_batch_size: 512 | global_step: 3 | reduced_train_loss: 11.03 | "
        "train_step_timing in s: 52.54\n"
        "Training epoch 0, iteration 4/4 | lr: 7.496e-07 | consumed_samples: 2560 | "
        "global_batch_size: 512 | global_step: 4 | reduced_train_loss: 11.03 | "
        "train_step_timing in s: 53.16\n"
    )

    (test_dir / "stdout.txt").write_text(stdout_content)

    return test_dir


def test_nemo_can_handle_directory(nemo_test_environment: Path) -> None:
    strategy = NeMoRunReportGenerationStrategy()
    assert strategy.can_handle_directory(nemo_test_environment)


def test_nemo_generate_report(nemo_test_environment: Path) -> None:
    strategy = NeMoRunReportGenerationStrategy()
    strategy.generate_report("nemo_test", nemo_test_environment)

    summary_file = nemo_test_environment / "summary.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_values = summary_file.read_text().strip().split(",")
    assert len(summary_values) == 4, "Summary file should contain four values (avg, median, min, max)."

    expected_values = [54.752, 53.16, 52.45, 61.94]
    actual_values = [float(value) for value in summary_values]

    assert pytest.approx(actual_values[0], 0.01) == expected_values[0], "Average value mismatch."
    assert pytest.approx(actual_values[1], 0.01) == expected_values[1], "Median value mismatch."
    assert pytest.approx(actual_values[2], 0.01) == expected_values[2], "Min value mismatch."
    assert pytest.approx(actual_values[3], 0.01) == expected_values[3], "Max value mismatch."
