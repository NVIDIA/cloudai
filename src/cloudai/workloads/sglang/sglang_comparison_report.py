# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import System, TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReportConfig
from cloudai.workloads.common.llm_serving_comparison_report import LLMServingComparisonReport

from .report_generation_strategy import SGLangBenchReportGenerationStrategy
from .sglang import SglangTestDefinition


class SGLangComparisonReport(LLMServingComparisonReport):
    """Comparison report for SGLang benchmark results."""

    def __init__(
        self, system: System, test_scenario: TestScenario, results_root: Path, config: ComparisonReportConfig
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "sglang_comparison.html"

    def can_handle(self, tr: TestRun) -> bool:
        return isinstance(tr.test, SglangTestDefinition)

    def parse_results(self, tr: TestRun):
        strategy = SGLangBenchReportGenerationStrategy(self.system, tr)
        results = strategy.parse_results()
        if results is None:
            return None
        return results, strategy.used_gpus_count()
