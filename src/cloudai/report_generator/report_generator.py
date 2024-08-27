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

import logging
import os

from cloudai import Test, TestScenario


class ReportGenerator:
    """
    Generates reports for each test in a TestScenario.

    By identifying the appropriate directories for each test and using test templates to generate detailed reports
    based on subdirectories.
    """

    def __init__(self, output_path: str) -> None:
        """
        Initialize the ReportGenerator with the path for output.

        Args:
            output_path (str): Output directory path.
        """
        self.output_path = output_path

    def generate_report(self, test_scenario: TestScenario) -> None:
        """
        Iterate over tests in the given test scenario.

        Identifies the relevant directories based on the test's section name, and generates a report for each test
        using its associated test template.

        Args:
            test_scenario (TestScenario): The scenario containing tests.
        """
        for test in test_scenario.tests:
            section_name = str(test.section_name) if test.section_name else ""
            if not section_name:
                logging.warning(f"Missing section name for test {test.name}")
                continue
            test_output_dir = os.path.join(self.output_path, section_name)
            if not os.path.exists(test_output_dir):
                logging.warning(f"Directory '{test_output_dir}' not found.")
                continue

            self._generate_test_report(test_output_dir, test)

    def _generate_test_report(self, directory_path: str, test: Test) -> None:
        """
        Generate reports for a test by iterating through subdirectories within the directory path.

        Checks if the test's template can handle each, and generating reports accordingly.

        Args:
            directory_path (str): Directory for the test's section.
            test (Test): The test for report generation.
        """
        for subdir in os.listdir(directory_path):
            subdir_path = os.path.join(directory_path, subdir)
            if os.path.isdir(subdir_path) and test.test_template.can_handle_directory(subdir_path):
                test.test_template.generate_report(test.name, subdir_path, test.sol)
            else:
                logging.warning(f"Skipping directory '{subdir_path}' for test '{test.name}'")
