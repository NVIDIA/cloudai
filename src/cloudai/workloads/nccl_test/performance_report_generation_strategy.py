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


from cloudai import System, TestRun

from .performance_report_generator import NcclTestPerformanceReportGenerator
from .report_generation_strategy import NcclTestReportGenerationStrategy


class NcclTestPerformanceReportGenerationStrategy(NcclTestReportGenerationStrategy):
    """Strategy for generating performance reports from NCCL test outputs."""

    def __init__(self, system: System, tr: TestRun) -> None:
        super().__init__(system, tr)
        self.performance_report = NcclTestPerformanceReportGenerator(self.test_run)

    def generate_report(self) -> None:
        self.performance_report.generate()
