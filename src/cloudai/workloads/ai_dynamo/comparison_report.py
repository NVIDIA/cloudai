# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib

import cloudai.core
import cloudai.report_generator.comparison_report
from cloudai.workloads.common.llm_serving_report import LLMServingComparisonReport

from .ai_dynamo import AIDynamoCmdArgs, AIDynamoTestDefinition
from .report_generation_strategy import AIDynamoReportGenerationStrategy


class AIDynamoComparisonReport(LLMServingComparisonReport):
    """Comparison report for AI Dynamo benchmark and accuracy results."""

    def __init__(
        self,
        system: cloudai.core.System,
        test_scenario: cloudai.core.TestScenario,
        results_root: pathlib.Path,
        config: cloudai.report_generator.comparison_report.ComparisonReportConfig,
    ) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.report_file_name = "ai_dynamo_comparison.html"

    def can_handle(self, tr: cloudai.core.TestRun) -> bool:
        return isinstance(tr.test, AIDynamoTestDefinition)

    def benchmark_cmd_args(self, tr: cloudai.core.TestRun) -> AIDynamoCmdArgs:
        if not isinstance(tr.test, AIDynamoTestDefinition):
            raise TypeError(f"{self.__class__.__name__} only supports AI Dynamo test runs.")
        return tr.test.cmd_args

    def comparison_values(self, tr: cloudai.core.TestRun) -> dict[str, object]:
        # AI Dynamo keeps both serving and benchmark settings in cmd_args, unlike
        # the standalone backends which have a separate bench_cmd_args model.
        return cloudai.report_generator.comparison_report.ComparisonReport.comparison_values(self, tr)

    def parse_results(self, tr: cloudai.core.TestRun):
        strategy = AIDynamoReportGenerationStrategy(self.system, tr)
        results = strategy.parse_results()
        if results is None:
            return None
        return results, strategy.used_gpus_count()

    def parse_accuracy(self, tr: cloudai.core.TestRun) -> float | None:
        return AIDynamoReportGenerationStrategy(self.system, tr).parse_accuracy()
