from typing import ClassVar

import pytest

from cloudai._core.test_scenario import TestRun
from cloudai.core import ReportGenerationStrategy, Test, TestDefinition, TestTemplate
from cloudai.models.workload import CmdArgs
from cloudai.systems.slurm import SlurmSystem


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
    def tr(self, slurm_system: SlurmSystem):
        return TestRun(
            name="test",
            test=Test(
                test_definition=TestDefinition(
                    name="test",
                    description="",
                    test_template_name="Test",
                    cmd_args=CmdArgs(),
                ),
                test_template=TestTemplate(slurm_system),
            ),
            num_nodes=1,
            nodes=[],
        )

    def test_no_reports(self, tr: TestRun):
        tr.reports.clear()
        tr.test.test_definition.agent_metrics = ["default"]
        assert tr.metric_reporter is None

    def test_no_metrics(self, tr: TestRun):
        tr.reports = {MyReport}
        tr.test.test_definition.agent_metrics = []
        assert tr.metric_reporter is None

    def test_one_report_multiple_metrics(self, tr: TestRun):
        tr.reports = {MyReport}
        metrics = ["m1", "m2"]
        MyReport.metrics = metrics
        tr.test.test_definition.agent_metrics = metrics

        assert tr.metric_reporter is MyReport

    def test_all_metrics_should_be_provided_by_the_same_reporter(self, tr: TestRun):
        tr.reports = {MyReport, MyReport2}
        metrics = ["m1", "m2"]
        MyReport.metrics = ["m1"]
        MyReport2.metrics = ["m2"]
        tr.test.test_definition.agent_metrics = metrics

        assert tr.metric_reporter is None
