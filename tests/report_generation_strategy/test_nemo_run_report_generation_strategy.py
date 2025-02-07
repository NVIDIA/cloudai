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
from unittest.mock import Mock

import pytest

from cloudai import Test, TestRun
from cloudai.schema.test_template.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy
from cloudai.test_definitions.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition


@pytest.fixture
def nemo_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="t",
            cmd_args=NeMoRunCmdArgs(docker_image_url="docker://url", task="task", recipe_name="recipe"),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

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

    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


def test_nemo_can_handle_directory(nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(nemo_tr)
    assert strategy.can_handle_directory()


def test_nemo_generate_report(nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 54.752,
        "Median": 53.16,
        "Min": 52.45,
        "Max": 61.94,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."
