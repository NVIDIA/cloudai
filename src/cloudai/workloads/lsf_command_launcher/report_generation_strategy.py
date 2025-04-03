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

from cloudai import ReportGenerationStrategy
from typing import ClassVar
from cloudai._core.test_scenario import METRIC_ERROR


class LSFCmdLauncherReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from LSF command launcher directories."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

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
    
    def get_metric(self, metric: str) -> float:
        if metric not in {"default", "step-time"}:
            return METRIC_ERROR

        return 0.0
    
    def generate_report(self) -> None:
        """Generate a report for the test run."""
        pass
