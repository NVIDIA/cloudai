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

import logging
import tarfile
from pathlib import Path

import jinja2

from cloudai import Reporter, TestRun
from cloudai.systems.slurm import SlurmSystem

from .common import ReportItem, SlurmReportItem


class StatusReporter(Reporter):
    """Generate a status report for a test scenario."""

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


class TarballReporter(Reporter):
    """Generate a tarball of the results directory."""

    def generate(self) -> None:
        self.load_test_runs()

        if any(not self.is_successful(tr) for tr in self.trs):
            self.create_tarball(self.results_root)

    def is_successful(self, tr: TestRun) -> bool:
        return tr.test.test_template.get_job_status(tr.output_path).is_successful

    def create_tarball(self, directory: Path) -> None:
        tarball_path = Path(str(directory) + ".tgz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(directory, arcname=directory.name)
        logging.info(f"Created tarball at {tarball_path}")
