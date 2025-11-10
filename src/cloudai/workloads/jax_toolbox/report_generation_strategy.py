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
import statistics
from typing import List

from cloudai.core import ReportGenerationStrategy


class JaxToolboxReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from JaxToolbox."""

    def can_handle_directory(self) -> bool:
        error_files = self.test_run.output_path.glob("error-*.txt")
        for error_file in error_files:
            with error_file.open("r") as file:
                content = file.read()
                if "[PAX STATUS]: E2E time: Elapsed time for <_main>: " in content:
                    return True
        return False

    def generate_report(self) -> None:
        times = self._extract_times()
        if times:
            stats = {
                "min": min(times),
                "max": max(times),
                "average": sum(times) / len(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            }
            self._write_report(stats)

    def _extract_times(self) -> List[float]:
        """
        Extract elapsed times from all error files matching the pattern in the directory.

        Starting after the 10th occurrence of a line matching the "[PAX STATUS]: train_step() took" pattern.

        Returns:
            List[float]: List of extracted times as floats, starting from the epoch after the 10th occurrence.
        """
        times = []
        error_files = self.test_run.output_path.glob("error-*.txt")
        for stderr_path in error_files:
            file_times = []
            epoch_count = 0
            with stderr_path.open("r") as file:
                for line in file:
                    if "[PAX STATUS]: train_step() took" in line:
                        epoch_count += 1
                        if epoch_count > 10:  # Start recording times after 10 epochs
                            # Extract the time value right after the keyword
                            parts = line.split("took")
                            time_str = parts[1].strip().split("seconds")[0].strip()
                            try:
                                time_value = float(time_str)
                                file_times.append(time_value)
                            except ValueError:
                                continue  # Skip any lines where conversion fails

            times.extend(file_times)

        if len(times) == 0:
            logging.warning(
                "JaxToolbox: The number of epochs is not sufficient to generate a report. "
                "At least 11 epochs are required to ensure accurate performance metrics, "
                "as the first 10 epochs are ignored due to overhead. Please run the tests for more than 10 epochs."
            )

        return times

    def _write_report(self, stats: dict) -> None:
        """
        Write the computed statistics to a file named 'report.txt' in the same directory.

        Args:
            stats (dict): Dictionary containing computed statistics.
        """
        report_path = self.test_run.output_path / "report.txt"
        with report_path.open("w") as file:
            for key, value in stats.items():
                file.write(f"{key.capitalize()}: {value}\n")
