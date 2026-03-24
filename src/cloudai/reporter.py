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
from pathlib import Path

import jinja2
from rich import box
from rich.console import Console
from rich.table import Table

from cloudai.core import Reporter, TestRun
from cloudai.report_generator.status_report import (
    DSEReportBuilder,
    DSESummary,
    ReportItem,
    format_duration,
    format_float,
    format_money,
    format_percent,
)


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
    def templates_dir(self) -> Path:
        return Path(__file__).parent / "util"

    def generate(self) -> None:
        self.load_test_runs()

        dse_builder = DSEReportBuilder(self.system, self.results_root, self.trs)
        dse_summaries = dse_builder.build(self.test_scenario.test_runs)

        self.to_html(dse_summaries)
        self.to_console(dse_summaries)

    def to_html(self, dse_summaries: list[DSESummary]) -> None:
        jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.templates_dir))
        template = jinja_env.get_template("general-report.jinja2")

        report_items = ReportItem.from_test_runs(self.trs, self.results_root)
        report = template.render(
            name=self.test_scenario.name,
            report_items=report_items,
            dse_summaries=dse_summaries,
            format_duration=format_duration,
            format_float=format_float,
            format_percent=format_percent,
            format_money=format_money,
        )
        report_path = self.results_root / f"{self.test_scenario.name}.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info(f"Generated scenario report at {report_path}")

    def to_console(self, dse_summaries: list[DSESummary]):
        if not self.trs:
            logging.debug("No test runs found, skipping summary.")
            return

        table = Table(title="Scenario results", title_justify="left", show_lines=True, box=box.DOUBLE_EDGE)
        for col in ["Case", "Status", "Details"]:
            table.add_column(col, overflow="fold")

        if dse_summaries:
            self._add_dse_rows(dse_summaries, table)
        else:
            self._add_standard_rows(table)

        console = Console()
        with console.capture() as capture:
            console.print(table)

        logging.info(capture.get())

    @staticmethod
    def _add_dse_rows(dse_summaries: list[DSESummary], table: Table):
        for summary in dse_summaries:
            details = [
                f"steps={summary.executed_steps}/{summary.total_space}",
                f"best_step={summary.best_step}",
                f"best_reward={format_float(summary.best_reward, 4)}",
                f"failures={summary.failure_count}",
            ]
            if summary.best_config_rel_path:
                details.append(summary.best_config_rel_path)
            table.add_row(summary.description, f"[bold]{summary.status_style}[/bold]", "\n".join(details))

    def _add_standard_rows(self, table: Table):
        for tr in self.trs:
            tr_status = tr.test.was_run_successful(tr)
            sts_text = f"[bold]{'[green]PASSED[/green]' if tr_status.is_successful else '[red]FAILED[/red]'}[/bold]"
            display_path = str(tr.output_path.absolute())
            with contextlib.suppress(ValueError):
                display_path = str(tr.output_path.absolute().relative_to(Path.cwd()))
            details_text = f"\n{tr_status.error_message}" if tr_status.error_message else ""
            table.add_row(tr.name, sts_text, f"{display_path}{details_text}")


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
