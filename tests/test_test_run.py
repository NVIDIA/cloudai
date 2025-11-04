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

from typing import ClassVar

import pytest

from cloudai._core.test_scenario import TestRun
from cloudai.core import ReportGenerationStrategy, TestDefinition
from cloudai.models.workload import CmdArgs


class MyReport(ReportGenerationStrategy):
    metrics: ClassVar[list[str]] = ["default", "metric1"]

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        return


class MyReport2(ReportGenerationStrategy):
    metrics: ClassVar[list[str]] = ["default", "metric2"]

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        return


class TestMetricsReporter:
    @pytest.fixture
    def tr(self) -> TestRun:
        return TestRun(
            name="test",
            test=TestDefinition(name="test", description="", test_template_name="Test", cmd_args=CmdArgs()),
            num_nodes=1,
            nodes=[],
        )

    def test_no_reports(self, tr: TestRun):
        tr.reports.clear()
        tr.test.agent_metrics = ["default"]
        assert tr.metric_reporter is None

    def test_no_metrics(self, tr: TestRun):
        tr.reports = {MyReport}
        tr.test.agent_metrics = []
        assert tr.metric_reporter is None

    def test_one_report_multiple_metrics(self, tr: TestRun):
        tr.reports = {MyReport}
        metrics = ["m1", "m2"]
        MyReport.metrics = metrics
        tr.test.agent_metrics = metrics

        assert tr.metric_reporter is MyReport

    def test_all_metrics_should_be_provided_by_the_same_reporter(self, tr: TestRun):
        tr.reports = {MyReport, MyReport2}
        metrics = ["m1", "m2"]
        MyReport.metrics = ["m1"]
        MyReport2.metrics = ["m2"]
        tr.test.agent_metrics = metrics

        assert tr.metric_reporter is None
