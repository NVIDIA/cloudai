# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jinja2
import toml

from ..systems.slurm.metadata import SlurmSystemMetadata
from ..systems.slurm.slurm_system import SlurmSystem
from .system import System
from .test_scenario import TestRun, TestScenario


def case_name(tr: TestRun) -> str:
    name = tr.name
    if tr.current_iteration > 0:
        name = f"{name} iter={tr.current_iteration}"
    if tr.step > 0:
        name = f"{name} step={tr.step}"
    return name


@dataclass
class ReportItem:
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
    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None

    @classmethod
    def get_metadata(cls, run_dir: Path) -> Optional[SlurmSystemMetadata]:
        if (run_dir / "metadata").exists():
            node_files = list(run_dir.glob("metadata/node-*.toml"))
            if node_files:
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
            if metadata := cls.get_metadata(tr.output_path):
                ri.nodes = metadata.slurm.node_list
            report_items.append(ri)

        return report_items


class Reporter(ABC):
    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path) -> None:
        self.system = system
        self.test_scenario = test_scenario
        self.results_root = results_root
        self.trs: list[TestRun] = []

    def load_test_runs(self):
        for tr in self.test_scenario.test_runs:
            tr_root = self.results_root / tr.name
            iters = list(subdir for subdir in tr_root.glob("*") if subdir.is_dir())
            for iter in sorted(iters):
                if tr.test.test_definition.is_dse_job:
                    steps = list(subdir for subdir in iter.glob("*") if subdir.is_dir())
                    for step in sorted(steps):
                        tr.current_iteration = int(iter.name)
                        tr.step = int(step.name)
                        tr.output_path = tr_root / f"{tr.current_iteration}" / f"{tr.step}"
                        self.trs.append(copy.deepcopy(tr))
                else:
                    tr.current_iteration = int(iter.name)
                    tr.step = 0
                    tr.output_path = tr_root / f"{tr.current_iteration}"
                    self.trs.append(copy.deepcopy(tr))

    @abstractmethod
    def generate(self) -> None: ...


class PerTestReporter(Reporter):
    def generate(self) -> None:
        self.load_test_runs()

        for tr in self.trs:
            test_output_dir = self.results_root / tr.name
            if not test_output_dir.exists() or not test_output_dir.is_dir():
                logging.warning(f"Directory '{test_output_dir}' not found.")
                continue

            self.generate_per_case_reports(test_output_dir, tr)

    def generate_per_case_reports(self, directory_path: Path, tr: TestRun) -> None:
        logging.debug(f"Available reports: {tr.reports} for directory: {directory_path}")
        for reporter in tr.reports:
            rgs = reporter(self.system, tr)

            for subdir in directory_path.iterdir():
                if tr.step > 0:
                    subdir = subdir / f"{tr.step}"
                tr.output_path = subdir

                if not rgs.can_handle_directory():
                    logging.warning(f"Skipping '{tr.output_path}', can't handle with " f"strategy={reporter.__name__}.")
                    continue
                try:
                    rgs.generate_report()
                except Exception as e:
                    logging.warning(
                        f"Error generating report for '{tr.output_path}' with strategy={reporter.__name__}: {e}"
                    )


class StatusReporter(Reporter):
    @property
    def template_file(self) -> str:
        if isinstance(self.system, SlurmSystem):
            return "general-slurm-report.jinja2"
        return "general-report.jinja2"

    def generate(self) -> None:
        self.load_test_runs()
        self.generate_scenario_report()

    def generate_scenario_report(self) -> None:
        template = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent.parent / "util")
        ).get_template(self.template_file)

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
