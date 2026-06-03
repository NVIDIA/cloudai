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

import pathlib

import cloudai.core
import cloudai.report_generator.comparison_report
from cloudai.workloads.common.llm_serving_report import LLMServingComparisonReport
from cloudai.workloads.sglang.report_generation_strategy import SGLangBenchReportGenerationStrategy
from cloudai.workloads.sglang.sglang import SglangBenchCmdArgs, SglangTestDefinition


class SGLangComparisonReport(LLMServingComparisonReport):
    """Comparison report for SGLang benchmark results."""

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "sglang_comparison.html"

    def can_handle(self, tr: cloudai.core.TestRun) -> bool:
        return isinstance(tr.test, SglangTestDefinition)

    def benchmark_cmd_args(self, tr: cloudai.core.TestRun) -> SglangBenchCmdArgs:
        if not isinstance(tr.test, SglangTestDefinition):
            raise TypeError(f"{self.__class__.__name__} only supports SGLang test runs.")
        return tr.test.bench_cmd_args

    def parse_results(self, tr: cloudai.core.TestRun):
        strategy = SGLangBenchReportGenerationStrategy(self.system, tr)
        results = strategy.parse_results()
        if results is None:
            return None
        return results, strategy.used_gpus_count()

    def parse_accuracy(self, tr: cloudai.core.TestRun) -> float | None:
        return SGLangBenchReportGenerationStrategy(self.system, tr).parse_semantic_accuracy()
