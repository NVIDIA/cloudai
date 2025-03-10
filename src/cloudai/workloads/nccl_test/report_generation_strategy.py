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

from cloudai import ReportGenerationStrategy, System, TestRun

from .performance_report_generator import NcclTestPerformanceReportGenerator
from .prediction_report_generator import NcclTestPredictionReportGenerator


class NcclTestReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NCCL test outputs."""

    def __init__(self, system: System, tr: TestRun) -> None:
        super().__init__(system, tr)

        self.performance_report = NcclTestPerformanceReportGenerator(
            self.test_run.output_path, self.test_run.name, self.test_run.sol
        )

        collective_type = self._normalize_collective_type(tr.test.test_definition.cmd_args.subtest_name)
        self.prediction_report = NcclTestPredictionReportGenerator(
            collective_type, self.test_run.output_path, tr.test.test_definition
        )

    def _normalize_collective_type(self, subtest_name: str) -> str:
        return subtest_name.replace("_perf", "").replace("_mpi", "")

    def can_handle_directory(self) -> bool:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                content = file.read()
                return bool(
                    re.search(r"out-of-place|in-place", content)
                    and re.search(
                        r"\b(size\s+count\s+type\s+redop\s+root\s+"
                        r"time\s+algbw\s+busbw\s+#wrong\s+time\s+"
                        r"algbw\s+busbw\s+#wrong)\b",
                        content,
                        re.IGNORECASE,
                    )
                )
        return False

    def generate_report(self) -> None:
        self.performance_report.generate()
        self.prediction_report.generate()
