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
import json
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

from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.report_generator.util import load_system_metadata
from cloudai.util.lazy_imports import lazy

from .core import METRIC_ERROR, CommandGenStrategy, Reporter, TestRun, case_name
from .models.scenario import TestRunDetails


@dataclass
class ReportItem:
    """Enhanced report item for Slurm systems with node information."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            ri = ReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := load_system_metadata(tr.output_path, results_root):
                ri.nodes = metadata.slurm.node_list
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

        report_items = ReportItem.from_test_runs(self.trs, self.results_root)
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


class SummaryReporter(Reporter):
    """Generate a machine-readable scenario summary for automation."""

    SUMMARY_FILE_NAME = "cloudai-summary.json"

    def generate(self) -> None:
        self.load_test_runs()
        report_path = self.results_root / self.SUMMARY_FILE_NAME
        with report_path.open("w") as f:
            json.dump(self.build_summary(), f, indent=2)
            f.write("\n")

        logging.info("Generated scenario summary at %s", report_path)

    def build_summary(self) -> dict[str, Any]:
        test_runs = self._test_runs_summary()
        return {
            "scenario": self.test_scenario.name,
            "status": self._scenario_status(test_runs),
            "result_dir": self._relative_path(self.results_root),
            "reports": self._scenario_artifacts(),
            "test_runs": test_runs,
        }

    def _scenario_status(self, test_runs: list[dict[str, Any]]) -> str:
        if not test_runs:
            return "unknown"
        if all(tr["status"] == "completed" for tr in test_runs):
            return "completed"
        return "failed"

    def _test_runs_summary(self) -> list[dict[str, Any]]:
        loaded_by_name: dict[str, list[TestRun]] = {}
        for tr in self.trs:
            loaded_by_name.setdefault(tr.name, []).append(tr)

        summary: list[dict[str, Any]] = []
        for test_run in self.test_scenario.test_runs:
            loaded_runs = loaded_by_name.get(test_run.name, [])
            if test_run.is_dse_job:
                summary.append(self._sweep_test_run_summary(test_run, loaded_runs))
            else:
                summary.extend(self._test_run_summary(tr) for tr in loaded_runs)

        return summary

    def _sweep_test_run_summary(self, tr: TestRun, sweeps: list[TestRun]) -> dict[str, Any]:
        sweep_summaries = [self._test_run_summary(sweep) for sweep in sweeps]
        summary = {
            "name": tr.name,
            "status": self._scenario_status(sweep_summaries),
            "output_path": self._relative_path(self.results_root / tr.name),
            "artifacts": self._artifacts_excluding(
                self.results_root / tr.name, [sweep.output_path for sweep in sweeps]
            ),
            "metrics": {},
            "sweeps": sweep_summaries,
        }
        return summary

    def _test_run_summary(self, tr: TestRun) -> dict[str, Any]:
        status = tr.test.was_run_successful(tr)
        summary = {
            "name": case_name(tr),
            "status": "completed" if status.is_successful else "failed",
            "output_path": self._relative_path(tr.output_path),
            "artifacts": self._artifacts(tr.output_path),
            "metrics": self._metrics(tr),
        }
        if status.error_message:
            summary["error_message"] = status.error_message
        return summary

    def _metrics(self, tr: TestRun) -> dict[str, float]:
        metrics = {}
        for metric in tr.test.agent_metrics:
            value = tr.get_metric_value(self.system, metric)
            if value is METRIC_ERROR:
                continue
            metrics[metric] = float(value)

        return metrics

    def _scenario_artifacts(self) -> list[dict[str, str]]:
        if not self.results_root.is_dir():
            return []

        return [
            self._artifact(path)
            for path in sorted(self.results_root.iterdir())
            if path.is_file() and path.name != self.SUMMARY_FILE_NAME
        ]

    def _artifacts(self, root: Path) -> list[dict[str, str]]:
        if not root.is_dir():
            return []

        return [self._artifact(path) for path in sorted(root.rglob("*")) if path.is_file()]

    def _artifacts_excluding(self, root: Path, excluded_roots: list[Path]) -> list[dict[str, str]]:
        if not root.is_dir():
            return []

        return [
            self._artifact(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and not any(self._is_relative_to(path, excluded_root) for excluded_root in excluded_roots)
        ]

    def _is_relative_to(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _artifact(self, path: Path) -> dict[str, str]:
        return {
            "path": self._relative_path(path),
            "format": path.suffix.removeprefix(".") or "unknown",
        }

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.results_root))
        except ValueError:
            return str(path)


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
