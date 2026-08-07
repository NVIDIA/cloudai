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

from __future__ import annotations

from pathlib import Path

import cloudai.metrics
from cloudai.core import System, TestScenario
from cloudai.report_generator.comparison_report import (
    ComparisonReportConfig,
    ComparisonSection,
)
from cloudai.report_generator.comparison_report_metric import MetricComparisonReport
from cloudai.report_generator.groups import GroupedTestRuns

from .nixl_bench import NIXLBenchTestDefinition


class NIXLBenchComparisonReport(MetricComparisonReport):
    """Comparison report for NIXL Bench."""

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "nixl_comparison.html"

    def load_test_runs(self):
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test, NIXLBenchTestDefinition)]

    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
        sections = []
        for group in cmp_groups:
            for metric in (cloudai.metrics.LATENCY, cloudai.metrics.BANDWIDTH):
                section = self.build_metric_section(
                    group,
                    metric,
                    x_dimension=cloudai.metrics.SIZE_BYTES.key,
                )
                if section is not None:
                    sections.append(section)
        return sections
