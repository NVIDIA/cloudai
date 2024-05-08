# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os
import statistics
from typing import List, Optional

from cloudai.schema.core.strategy import (
    ReportGenerationStrategy,
    StrategyRegistry,
)
from cloudai.schema.system import SlurmSystem

from .template import JaxToolbox


@StrategyRegistry.strategy(ReportGenerationStrategy, [SlurmSystem], [JaxToolbox])
class JaxToolboxReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from JaxToolbox.
    """

    def can_handle_directory(self, directory_path: str) -> bool:
        error_files = glob.glob(os.path.join(directory_path, "error-*.txt"))
        for error_file in error_files:
            with open(error_file, "r") as file:
                content = file.read()
                if "[PAX STATUS]: E2E time: Elapsed time for <_main>: " in content:
                    return True
        return False

    def generate_report(self, directory_path: str, sol: Optional[float] = None) -> None:
        times = self._extract_times(directory_path)
        if times:
            stats = {
                "min": min(times),
                "max": max(times),
                "average": sum(times) / len(times),
                "median": statistics.median(times),
            }
            self._write_report(directory_path, stats)

    def _extract_times(self, directory_path: str) -> List[float]:
        """
        Extracts elapsed times from all error files matching the pattern in the directory,
        excluding the first time value recorded in each file.

        Args:
            directory_path (str): Directory containing error files.

        Returns:
            List[float]: List of extracted times as floats, after excluding the first time from each file.
        """
        times = []
        error_files = glob.glob(os.path.join(directory_path, "error-*.txt"))
        for stderr_path in error_files:
            file_times = []
            with open(stderr_path, "r") as file:
                for line in file:
                    if "Elapsed time for" in line and "run" in line and ":434" in line:
                        parts = line.split()
                        time_str = parts[parts.index("<run>:") + 1]
                        try:
                            time_value = float(time_str.split("seconds")[0])
                            file_times.append(time_value)
                        except ValueError:
                            continue  # Skip any lines where conversion fails

            # Exclude the first time record from each file if it exists
            if file_times:
                times.extend(file_times[1:])

        return times

    def _write_report(self, directory_path: str, stats: dict) -> None:
        """
        Writes the computed statistics to a file named 'report.txt' in the
        same directory.

        Args:
            directory_path (str): Path to the directory.
            stats (dict): Dictionary containing computed statistics.
        """
        report_path = os.path.join(directory_path, "report.txt")
        with open(report_path, "w") as file:
            for key, value in stats.items():
                file.write(f"{key.capitalize()}: {value}\n")
