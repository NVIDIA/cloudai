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
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jinja2
import toml
from rich import box
from rich.console import Console
from rich.table import Table

from cloudai.report_generator.dse_report import build_dse_summaries, load_trajectory_dataframe
from cloudai.report_generator.util import load_system_metadata

from .core import CommandGenStrategy, Reporter, TestRun, case_name
from .models.scenario import TestRunDetails


@dataclass
class ReportItem:
    """Enhanced report item for Slurm systems with node information."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None
    is_successful: Optional[bool] = None
    error_message: str = ""

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            ri = ReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := load_system_metadata(tr.output_path, results_root):
                ri.nodes = metadata.slurm.node_list
            status = tr.test.was_run_successful(tr)
            ri.is_successful = status.is_successful
            ri.error_message = status.error_message
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


class JUnitReporter(Reporter):
    """Generate a JUnit XML report for all scenario test executions."""

    REPORT_FILE_NAME = "junit.xml"

    def generate(self) -> None:
        self.load_test_runs()

        results = [(tr, tr.test.was_run_successful(tr), self._duration(tr.output_path)) for tr in self.trs]
        failures = sum(not status.is_successful for _, status, _ in results)
        durations = [duration for _, _, duration in results if duration is not None]

        suite_attributes = {
            "name": self.test_scenario.name,
            "tests": str(len(results)),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
        }
        if durations:
            suite_attributes["time"] = self._format_duration(sum(durations))

        root = ET.Element("testsuites", suite_attributes)
        suite = ET.SubElement(root, "testsuite", suite_attributes)
        for tr, status, duration in results:
            attributes = {"name": case_name(tr), "classname": self.test_scenario.name}
            if duration is not None:
                attributes["time"] = self._format_duration(duration)

            testcase = ET.SubElement(suite, "testcase", attributes)
            if not status.is_successful:
                message = status.error_message or "Test run failed"
                failure = ET.SubElement(testcase, "failure", {"message": self._xml_text(message)})
                failure.text = self._xml_text(message)

            self._add_log(testcase, "system-out", tr.output_path / "stdout.txt")
            self._add_log(testcase, "system-err", tr.output_path / "stderr.txt")

        ET.indent(root)
        report_path = self.results_root / self.REPORT_FILE_NAME
        ET.ElementTree(root).write(report_path, encoding="utf-8", xml_declaration=True)
        logging.info("Generated JUnit report at %s", report_path)

    @staticmethod
    def _duration(output_path: Path) -> float | None:
        # Duration is currently available only for Slurm workloads, which persist slurm-job.toml.
        metadata_path = output_path / "slurm-job.toml"
        if not metadata_path.is_file():
            return None
        try:
            duration = toml.load(metadata_path).get("elapsed_time_sec")
            return float(duration) if duration is not None else None
        except (OSError, TypeError, ValueError, toml.TomlDecodeError) as exc:
            logging.debug("Could not read execution duration from %s: %s", metadata_path, exc)
            return None

    @staticmethod
    def _format_duration(duration: float) -> str:
        return f"{duration:.3f}"

    @classmethod
    def _add_log(cls, testcase: ET.Element, tag: str, path: Path) -> None:
        if not path.is_file():
            return
        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            logging.debug("Could not read test log %s: %s", path, exc)
            return
        ET.SubElement(testcase, tag).text = cls._xml_text(content)

    @staticmethod
    def _xml_text(value: str) -> str:
        """Remove control characters that XML 1.0 cannot represent."""
        return "".join(
            char
            for char in value
            if char in "\t\n\r"
            or "\u0020" <= char <= "\ud7ff"
            or "\ue000" <= char <= "\ufffd"
            or "\U00010000" <= char <= "\U0010ffff"
        )


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
            loaded_trajectory = load_trajectory_dataframe(tr_root)
            if loaded_trajectory is None:
                logging.warning("No trajectory file found for %s in %s", tr.name, tr_root)
                continue

            _, df = loaded_trajectory
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
