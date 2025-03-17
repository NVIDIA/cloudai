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

import logging
from pathlib import Path

import jinja2

from .system import System
from .test_scenario import TestRun, TestScenario


class Reporter:
    """
    Generates reports for each test in a TestScenario.

    By identifying the appropriate directories for each test and using test templates to generate detailed reports
    based on subdirectories.
    """

    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path) -> None:
        self.system = system
        self.test_scenario = test_scenario
        self.results_root = results_root

    def generate(self) -> None:
        """
        Iterate over tests in the given test scenario.

        Identifies the relevant directories based on the test's section name, and generates a report for each test
        using its associated test template.

        Args:
            test_scenario (TestScenario): The scenario containing tests.
        """
        self.generate_scenario_report()

        for tr in self.test_scenario.test_runs:
            test_output_dir = self.results_root / tr.name
            if not test_output_dir.exists() or not test_output_dir.is_dir():
                logging.warning(f"Directory '{test_output_dir}' not found.")
                continue

            self.generate_per_case_reports(test_output_dir, tr)

    def generate_scenario_report(self) -> None:
        template = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent.parent / "util")
        ).get_template("general-report.jinja2")

        results = {}
        for tr in self.test_scenario.test_runs:
            for iter in range(tr.iterations):
                run_dir = self.results_root / tr.name / f"{iter}"
                if run_dir.exists():
                    results.setdefault(
                        tr.name + f"{iter}", {"logs_path": f"./{run_dir.relative_to(self.results_root)}"}
                    )

        report = template.render(test_scenario=self.test_scenario, tr_results=results)
        report_path = self.results_root / f"{self.test_scenario.name}.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info(f"Generated scenario report at {report_path}")

    def generate_per_case_reports(self, directory_path: Path, tr: TestRun) -> None:
        """
        Generate reports for a test by iterating through subdirectories within the directory path.

        Checks if the test's template can handle each, and generating reports accordingly.

        Args:
            directory_path (Path): Directory for the test's section.
            tr (TestRun): The test run object.
        """
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
