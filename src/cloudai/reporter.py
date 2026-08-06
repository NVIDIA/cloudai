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

import contextlib
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jinja2
import toml
from rich import box
from rich.console import Console
from rich.table import Table

import cloudai.metrics
from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.report_generator.util import load_system_metadata
from cloudai.util.lazy_imports import lazy

from .core import CommandGenStrategy, Reporter, System, TestRun, case_name
from .models.scenario import TestRunDetails


@dataclass
class SOLMetricReport:
    """Presentation-ready SOL details for one metric in one test run."""

    key: str
    display_name: str
    unit: str
    coverage_text: str
    worst: str
    median: str
    best: str
    coordinate_headers: list[str]
    rows: list[dict[str, Any]]
    chart: dict[str, Any] | None


def _build_metric_chart(
    metric: cloudai.metrics.MetricDefinition,
    assessments: list[cloudai.metrics.MetricAssessment],
    chart_id: str,
) -> dict[str, Any] | None:
    rows = [{**assessment.observation.dimensions, "assessment": assessment} for assessment in assessments]
    df = lazy.pd.DataFrame(rows)
    view = cloudai.metrics.build_metric_view(metric, [assessments])
    if view is None or view.x_dimension is None:
        return None

    x_dimension = view.x_dimension
    series_dimensions = list(view.series_dimensions)
    grouped = df.groupby(series_dimensions, dropna=False, sort=False) if series_dimensions else [((), df)]
    x_values = sorted(df[x_dimension].unique())

    datasets: list[dict[str, Any]] = []
    sol_datasets: dict[tuple[float | None, ...], dict[str, Any]] = {}
    for series_idx, (series_key, series_df) in enumerate(grouped):
        if not isinstance(series_key, tuple):
            series_key = (series_key,)
        label = (
            " · ".join(
                f"{cloudai.metrics.dimension_label(name)}={cloudai.metrics.format_dimension(name, value)}"
                for name, value in zip(series_dimensions, series_key, strict=True)
            )
            or "Measured"
        )
        point_by_x = {row[x_dimension]: row.assessment for _, row in series_df.iterrows()}
        datasets.append(
            {
                "label": label,
                "data": [point_by_x[x].observation.value if x in point_by_x else None for x in x_values],
                "source_color_index": series_idx,
            }
        )
        sol_data = tuple(point_by_x[x].sol if x in point_by_x else None for x in x_values)
        if not any(value is not None for value in sol_data):
            continue
        if existing := sol_datasets.get(sol_data):
            existing["label"] = "SOL"
            continue
        sol_dataset = {
            "label": f"{label} · SOL" if series_dimensions else "SOL",
            "data": list(sol_data),
            "is_sol": True,
        }
        sol_datasets[sol_data] = sol_dataset
        datasets.append(sol_dataset)

    return {
        "id": chart_id,
        "type": "line",
        "labels": [cloudai.metrics.format_dimension(x_dimension, value) for value in x_values],
        "datasets": datasets,
        "sol_color": "#741D9D",
        "x_axis_label": cloudai.metrics.dimension_label(x_dimension),
        "x_axis_type": "indexed_category",
        "y_axis_label": f"{metric.display_name} ({metric.unit})",
        "y_axis_type": "linear",
    }


def _build_sol_metric_reports(
    assessments: list[cloudai.metrics.MetricAssessment], item_idx: int
) -> list[SOLMetricReport]:
    grouped: dict[str, list[cloudai.metrics.MetricAssessment]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment.observation.metric.key, []).append(assessment)

    reports: list[SOLMetricReport] = []
    for metric_idx, metric_assessments in enumerate(grouped.values()):
        summary = cloudai.metrics.summarize_assessments(metric_assessments)[0]
        if summary.matched == 0:
            continue
        metric = summary.metric
        coordinate_headers = list(metric_assessments[0].observation.dimensions)
        rows = []
        for assessment in metric_assessments:
            dimensions = assessment.observation.dimensions
            selector = assessment.target.match if assessment.target is not None else None
            rows.append(
                {
                    "coordinates": [
                        cloudai.metrics.format_dimension(name, dimensions[name]) for name in coordinate_headers
                    ],
                    "measured": f"{assessment.observation.value:g}",
                    "sol": f"{assessment.sol:g}" if assessment.sol is not None else "n/a",
                    "attainment": f"{assessment.attainment:.1%}" if assessment.attainment is not None else "n/a",
                    "gap": f"{assessment.gap:+g}" if assessment.gap is not None else "n/a",
                    "target": (
                        ", ".join(
                            f"{cloudai.metrics.dimension_label(name)}={cloudai.metrics.format_dimension(name, value)}"
                            for name, value in selector.items()
                        )
                        if selector
                        else ("Default" if assessment.target is not None else "n/a")
                    ),
                }
            )

        coverage_text = (
            f"{summary.observations} measurements compared with SOL"
            if summary.matched == summary.observations
            else f"SOL available for {summary.matched} of {summary.observations} measurements"
        )
        reports.append(
            SOLMetricReport(
                key=metric.key,
                display_name=metric.display_name,
                unit=metric.unit,
                coverage_text=coverage_text,
                worst=f"{summary.worst_attainment:.1%}",
                median=f"{summary.median_attainment:.1%}",
                best=f"{summary.best_attainment:.1%}",
                coordinate_headers=[cloudai.metrics.dimension_label(name) for name in coordinate_headers],
                rows=rows,
                chart=_build_metric_chart(
                    metric,
                    metric_assessments,
                    f"sol-chart-{item_idx}-{metric_idx}",
                ),
            )
        )
    return reports


@dataclass
class ReportItem:
    """Enhanced report item for Slurm systems with node information."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None
    sol_summaries: list[cloudai.metrics.MetricAssessmentSummary] | None = None
    sol_metrics: list[SOLMetricReport] | None = None

    @classmethod
    def from_test_runs(
        cls, test_runs: list[TestRun], results_root: Path, system: System | None = None
    ) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for item_idx, tr in enumerate(test_runs):
            ri = ReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := load_system_metadata(tr.output_path, results_root):
                ri.nodes = metadata.slurm.node_list
            if system is not None:
                try:
                    assessments = cloudai.metrics.assess_test_run_metrics(system, tr)
                    ri.sol_summaries = [
                        summary for summary in cloudai.metrics.summarize_assessments(assessments) if summary.matched
                    ]
                    ri.sol_metrics = _build_sol_metric_reports(assessments, item_idx)
                except Exception as exc:
                    logging.warning("Failed to assess SOL metrics for '%s': %s", tr.output_path, exc)
            report_items.append(ri)

        return report_items


class PerTestReporter(Reporter):
    """Generates reports per test using test-specific reporting strategies."""

    def generate(self) -> None:
        self.load_test_runs()

        for tr in self.trs:
            logging.debug(f"Available reports: {[r.__name__ for r in tr.reports]} for directory: {tr.output_path}")
            for reporter in tr.reports:
                rgs = reporter(self.system, tr)

                if not rgs.can_handle_directory():
                    logging.warning(f"Skipping '{tr.output_path}', can't handle with strategy={reporter.__name__}.")
                    continue
                try:
                    rgs.generate_report()
                except Exception as e:
                    logging.warning(
                        f"Error generating report for '{tr.output_path}' with strategy={reporter.__name__}: {e}"
                    )


class StatusReporter(Reporter):
    """Generates HTML status reports with system-specific templates."""

    @property
    def template_file_path(self) -> Path:
        return Path(__file__).parent / "util"

    @property
    def template_file(self) -> str:
        return "general-report.jinja2"

    def generate(self) -> None:
        self.load_test_runs()
        self.generate_scenario_report()
        self.print_summary()

    def generate_scenario_report(self) -> None:
        template = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_file_path)).get_template(
            self.template_file
        )

        report_items = ReportItem.from_test_runs(self.trs, self.results_root, self.system)
        report = template.render(name=self.test_scenario.name, report_items=report_items)
        report_path = self.results_root / f"{self.test_scenario.name}.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info("Generated scenario report at %s", report_path)

    def print_summary(self) -> None:
        if not self.trs:
            logging.debug("No test runs found, skipping summary.")
            return

        table = Table(title="Scenario results", title_justify="left", show_lines=True, box=box.DOUBLE_EDGE)
        for col in ["Case", "Status", "Details"]:
            table.add_column(col, overflow="fold")

        for tr in self.trs:
            tr_status = tr.test.was_run_successful(tr)
            sts_text = f"[bold]{'[green]PASSED[/green]' if tr_status.is_successful else '[red]FAILED[/red]'}[/bold]"
            display_path = str(tr.output_path.absolute())
            with contextlib.suppress(ValueError):
                display_path = str(tr.output_path.absolute().relative_to(Path.cwd()))
            details_text = f"\n{tr_status.error_message}" if tr_status.error_message else ""
            columns = [tr.name, sts_text, f"{display_path}{details_text}"]
            table.add_row(*columns)

        console = Console()
        with console.capture() as capture:
            console.print(table)  # doesn't print to stdout, captures only

        logging.info(capture.get())

        sol_table = Table(title="Performance vs SOL", title_justify="left", show_lines=True, box=box.DOUBLE_EDGE)
        for col in ["Case", "Metric", "Coverage", "Worst", "Median", "Best"]:
            sol_table.add_column(col, overflow="fold")

        has_sol = False
        for tr in self.trs:
            try:
                summaries = cloudai.metrics.summarize_assessments(
                    cloudai.metrics.assess_test_run_metrics(self.system, tr)
                )
            except Exception as exc:
                logging.warning("Failed to assess SOL metrics for '%s': %s", tr.output_path, exc)
                continue
            for summary in summaries:
                if summary.matched == 0:
                    continue
                has_sol = True
                values = [summary.worst_attainment, summary.median_attainment, summary.best_attainment]
                sol_table.add_row(
                    case_name(tr),
                    summary.metric.display_name,
                    f"{summary.matched}/{summary.observations}",
                    *(f"{value:.1%}" if value is not None else "n/a" for value in values),
                )

        if has_sol:
            with console.capture() as capture:
                console.print(sol_table)
            logging.info(capture.get())


class DSEReporter(Reporter):
    """
    Generate DSE-specific scenario artifacts.

    For scenarios containing DSE test cases, this reporter produces:

    - a dedicated HTML report at `<results>/<scenario>-dse-report.html`
    - one best-config TOML per DSE test case iteration at
      `<results>/<dse-case>/<iteration>/<dse-case>.toml`
    """

    @property
    def templates_dir(self) -> Path:
        return Path(__file__).parent / "util"

    def generate(self) -> None:
        self.load_test_runs()

        dse_cases = build_dse_summaries(
            system=self.system,
            results_root=self.results_root,
            loaded_test_runs=self.trs,
            test_cases=self.test_scenario.test_runs,
        )

        if not dse_cases:
            return

        self.report_best_dse_config()

        jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.templates_dir))
        template = jinja_env.get_template("dse-report.jinja2")

        report = template.render(name=self.test_scenario.name, dse_cases=dse_cases)
        report_path = self.results_root / f"{self.test_scenario.name}-dse-report.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info(f"Generated scenario report at {report_path}")

    def report_best_dse_config(self):
        """Persist the highest-reward configuration for each DSE test case iteration."""
        for tr in self.test_scenario.test_runs:
            if not tr.test.is_dse_job:
                continue

            tr_root = self.results_root / tr.name / f"{tr.current_iteration}"
            trajectory_file = tr_root / "trajectory.csv"
            if not trajectory_file.is_file():
                logging.warning("No trajectory file found for %s at %s", tr.name, trajectory_file)
                continue

            df = lazy.pd.read_csv(trajectory_file)
            best_step = df.loc[df["reward"].idxmax()]["step"]
            best_step_details = tr_root / f"{best_step}" / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
            if not best_step_details.is_file():
                logging.warning("No best step found for %s at %s", tr.name, best_step_details)
                continue

            with best_step_details.open() as f:
                try:
                    trd = TestRunDetails.model_validate(toml.load(f))
                except Exception as exc:
                    logging.warning("Failed to validate test run for %s: %s", tr.name, exc, exc_info=True)
                    continue

            best_config_path = tr_root / f"{tr.name}.toml"
            logging.info("Writing best config for %s to %s", tr.name, best_config_path)
            with best_config_path.open("w") as f:
                toml.dump(trd.test_definition.model_dump(), f)


class TarballReporter(Reporter):
    """Creates tarballs of results for failed test runs."""

    def generate(self) -> None:
        self.load_test_runs()

        if any(not self.is_successful(tr) for tr in self.trs):
            self.create_tarball(self.results_root)

    def is_successful(self, tr: TestRun) -> bool:
        return tr.test.was_run_successful(tr).is_successful

    def create_tarball(self, directory: Path) -> None:
        tarball_path = Path(str(directory) + ".tgz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(directory, arcname=directory.name)
        logging.info(f"Created tarball at {tarball_path}")
