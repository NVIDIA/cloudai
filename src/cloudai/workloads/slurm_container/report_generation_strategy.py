# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re

from cloudai.core import ReportGenerationStrategy


class SlurmContainerReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for a generic Slurm container test."""

    def can_handle_directory(self) -> bool:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                if re.search(
                    r"Training epoch \d+, iteration \d+/\d+ | lr: [\d.]+ | global_batch_size: \d+ | global_step: \d+ | "
                    r"reduced_train_loss: [\d.]+ | train_step_timing in s: [\d.]+",
                    file.read(),
                ):
                    return True
        return False

    def generate_report(self) -> None:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if not stdout_path.is_file():
            return

        with stdout_path.open("r") as file:
            lines = file.readlines()
            with open(self.test_run.output_path / "report.csv", "w") as csv_file:
                csv_file.write(
                    "epoch,iteration,lr,global_batch_size,global_step,reduced_train_loss,train_step_timing,consumed_samples\n"
                )
                for line in lines:
                    pattern = (
                        r"Training epoch (\d+), iteration (\d+)/\d+ \| lr: ([\d.]+) \| global_batch_size: (\d+) \| "
                        r"global_step: (\d+) \| reduced_train_loss: ([\d.]+) \| train_step_timing in s: ([\d.]+)"
                    )
                    if " | consumed_samples:" in line:
                        pattern = (
                            r"Training epoch (\d+), iteration (\d+)/\d+ \| lr: ([\d.]+) \| global_batch_size: (\d+) \| "
                            r"global_step: (\d+) \| reduced_train_loss: ([\d.]+) \| train_step_timing in s: ([\d.]+) "
                            r"\| consumed_samples: (\d+)"
                        )

                    match = re.match(pattern, line)
                    if match:
                        csv_file.write(",".join(match.groups()) + "\n")
