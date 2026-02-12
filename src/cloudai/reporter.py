# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import jinja2
import toml
from rich import box
from rich.console import Console
from rich.table import Table

from cloudai.util.lazy_imports import lazy

from .core import CommandGenStrategy, Reporter, TestRun, case_name
from .models.scenario import TestRunDetails
from .systems.slurm import SlurmSystem, SlurmSystemMetadata


@dataclass
class ReportItem:
    """Basic report item for general systems."""

    name: str
    description: str
    logs_path: Optional[str] = None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            report_items.append(ReportItem(case_name(tr), tr.test.description))
            if tr.output_path.exists():
                report_items[-1].logs_path = f"./{tr.output_path.relative_to(results_root)}"
        return report_items


@dataclass
class SlurmReportItem:
    """Enhanced report item for Slurm systems with node information."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None

    @classmethod
    def get_metadata(cls, run_dir: Path, results_root: Path) -> Optional[SlurmSystemMetadata]:
        metadata_path = run_dir / "metadata"
        if not metadata_path.exists():
            logging.debug(f"No metadata folder found in {run_dir=}")
            if not (results_root / "metadata").exists():
                logging.debug(f"No metadata folder found in {results_root=}")
                return None
            else:  # single-sbatch case
                metadata_path = results_root / "metadata"

        node_files = list(metadata_path.glob("node-*.toml"))
        if not node_files:
            logging.debug(f"No node files found in {metadata_path}")
            return None

        node_file = node_files[0]
        with node_file.open() as f:
            try:
                return SlurmSystemMetadata.model_validate(toml.load(f))
            except Exception as e:
                logging.debug(f"Error validating metadata for {node_file}: {e}")

        return None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["SlurmReportItem"]:
        report_items: list[SlurmReportItem] = []
        for tr in test_runs:
            ri = SlurmReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := cls.get_metadata(tr.output_path, results_root):
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
        if isinstance(self.system, SlurmSystem):
            return "general-slurm-report.jinja2"
        return "general-report.jinja2"

    def best_dse_config_file_name(self, tr: TestRun) -> str:
        return f"{tr.name}.toml"

    def generate(self) -> None:
        self.load_test_runs()
        self.generate_scenario_report()
        self.report_best_dse_config()
        self.print_summary()

    def generate_scenario_report(self) -> None:
        template = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_file_path)).get_template(
            self.template_file
        )

        report_items = (
            SlurmReportItem.from_test_runs(self.trs, self.results_root)
            if isinstance(self.system, SlurmSystem)
            else ReportItem.from_test_runs(self.trs, self.results_root)
        )
        report = template.render(name=self.test_scenario.name, report_items=report_items)
        report_path = self.results_root / f"{self.test_scenario.name}.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info(f"Generated scenario report at {report_path}")

    def report_best_dse_config(self):
        for tr in self.test_scenario.test_runs:
            if not tr.test.is_dse_job:
                continue

            tr_root = self.results_root / tr.name / f"{tr.current_iteration}"
            trajectory_file = tr_root / "trajectory.csv"
            if not trajectory_file.exists():
                logging.warning(f"No trajectory file found for {tr.name} at {trajectory_file}")
                continue

            df = lazy.pd.read_csv(trajectory_file)
            best_step = df.loc[df["reward"].idxmax()]["step"]
            best_step_details = tr_root / f"{best_step}" / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
            with best_step_details.open() as f:
                trd = TestRunDetails.model_validate(toml.load(f))

            best_config_path = tr_root / self.best_dse_config_file_name(tr)
            logging.info(f"Writing best config for {tr.name} to {best_config_path}")
            with best_config_path.open("w") as f:
                toml.dump(trd.test_definition.model_dump(), f)

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
