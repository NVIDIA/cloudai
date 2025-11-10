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

from pathlib import Path

from cloudai.core import GradingStrategy


class UCCTestGradingStrategy(GradingStrategy):
    """
    Performance grading strategy for UCCTest test templates on Slurm systems.

    Evaluates the test's performance by comparing the maximum bus bandwidth achieved during the test against the test's
    ideal performance metric. The grade is normalized and scaled between 0 and 100.
    """

    def grade(self, directory_path: Path, ideal_perf: float) -> float:
        """
        Grade the performance of a UCCTest based on the maximum bus bandwidth.

        Reported in the test's stdout.txt file, considering both in-place and out-of-place updates.

        Args:
            directory_path (Path): Path to the directory containing the test's output.
            ideal_perf (float): The ideal performance value for comparison.

        Returns:
            float: The performance grade of the test, normalized and
                   scaled between 0 and 100.
        """
        stdout_path = directory_path / "stdout.txt"
        if not stdout_path.is_file():
            return 0.0

        max_bus_bw = self._extract_max_bus_bandwidth(stdout_path)
        if max_bus_bw is None or ideal_perf <= 0:
            return 0.0

        normalized_perf = (max_bus_bw / ideal_perf) * 100
        grade = min(max(normalized_perf, 0), 100)
        return grade

    def _extract_max_bus_bandwidth(self, stdout_path: Path) -> float:
        """
        Extract the maximum bus bandwidth from the UCCTest output file.

        Args:
            stdout_path (Path): Path to the stdout.txt file containing the UCCTest output.

        Returns:
            float: The maximum bus bandwidth value.
        """
        max_bus_bw = 0.0
        with stdout_path.open("r") as file:
            for line in file:
                parts = line.split()
                if len(parts) == 8:  # Ensure it's a data line
                    bw = float(parts[5])
                    max_bus_bw = max(max_bus_bw, bw)
        return max_bus_bw
