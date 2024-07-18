# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import logging
import os
from typing import Dict, List

from .test import Test
from .test_scenario import TestScenario


class Grader:
    """
    Class responsible for grading the performance of tests within a test scenario and generating a report.

    Attributes
        output_path (str): The path where the performance results are stored.
        logger (logging.Logger): Logger for the class, used to log messages related to the grading process.
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def grade(self, test_scenario: TestScenario) -> str:
        """
        Perform grading based on performance metrics.

        Done for each test in the given test scenario, considering the weight of each test, and
        generates a comprehensive weighted report.

        Args:
            test_scenario (TestScenario): The test scenario containing multiple tests to grade.

        Returns:
            str: A report summarizing the weighted performance grades.
        """
        weighted_perfs: List[float] = []
        test_perfs: Dict[str, List[float]] = {}
        total_weight = sum(test.weight for test in test_scenario.tests)

        for test in test_scenario.tests:
            section_name = str(test.section_name) if test.section_name else ""
            if not section_name:
                logging.warning(f"Missing section name for test {test.name}")
                continue
            test_output_dir = os.path.join(self.output_path, section_name)
            perfs = self._get_perfs_from_subdirs(test_output_dir, test)
            avg_perf = sum(perfs) / len(perfs) if perfs else 0
            test_perfs[test.name] = perfs + [avg_perf]
            weighted_avg = (avg_perf * test.weight / total_weight) if total_weight else 0
            weighted_perfs.append(weighted_avg)

        overall_weighted_avg = sum(weighted_perfs)
        report = self._generate_report(test_perfs, overall_weighted_avg)
        self._save_report(report)
        return report

    def _get_perfs_from_subdirs(self, directory_path: str, test: Test) -> List[float]:
        """
        Average performance values from subdirectories within a given path, according to the test's grading template.

        Args:
            directory_path (str): Directory path.
            test (Test): The test to grade.

        Returns:
            List[float]: A list of performance values.
        """
        perfs = []
        for subdir in os.listdir(directory_path):
            if subdir.isdigit():
                subdir_path = os.path.join(directory_path, subdir)
                if os.path.isdir(subdir_path):
                    perf = test.test_template.grade(subdir_path, test.ideal_perf)
                    perfs.append(perf)
        return perfs

    def _generate_report(self, test_perfs: Dict[str, List[float]], overall_avg: float) -> str:
        """
        Generate a human-readable report from test performance metrics.

        Args:
            test_perfs (Dict[str, List[float]]): The performance metrics for each test.
            overall_avg (float): The overall average performance.

        Returns:
            str: The generated report.
        """
        report_lines = ["Test Performance Report:"]
        for test, perfs in test_perfs.items():
            report_lines.append(f"{test}: Min: {min(perfs[:-1])}, " f"Max: {max(perfs[:-1])}, " f"Avg: {perfs[-1]}")
        report_lines.append(f"Overall Average Performance: {overall_avg}")
        return "\n".join(report_lines)

    def _save_report(self, report: str) -> None:
        """
        Save the generated report to a CSV file at the output path.

        Args:
            report (str): The report to save.
        """
        report_path = os.path.join(self.output_path, "performance_report.csv")
        with open(report_path, "w", newline="") as file:
            writer = csv.writer(file)
            for line in report.split("\n"):
                writer.writerow([line])
