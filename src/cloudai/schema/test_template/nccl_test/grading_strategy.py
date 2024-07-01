#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from cloudai import GradingStrategy


class NcclTestGradingStrategy(GradingStrategy):
    """
    Performance grading strategy for NcclTest test templates on Slurm systems.

    Evaluates the test's performance by comparing the maximum bus bandwidth achieved during the test against the test's
    ideal performance metric. The grade is normalized and scaled between 0 and 100.
    """

    def grade(self, directory_path: str, ideal_perf: float) -> float:
        """
        Gradesthe performance of an NcclTest based on the maximum bus bandwidth.

        Reported in the test's stdout.txt file, considering both in-place and out-of-place updates.

        Args:
            directory_path (str): Path to the directory containing the
                                  test's output.
            ideal_perf (float): The ideal performance value for comparison.

        Returns:
            float: The performance grade of the test, normalized and
                   scaled between 0 and 100.
        """
        stdout_path = os.path.join(directory_path, "stdout.txt")
        if not os.path.isfile(stdout_path):
            return 0.0

        max_bus_bw = self._extract_max_bus_bandwidth(stdout_path)
        if max_bus_bw is None or ideal_perf <= 0:
            return 0.0

        normalized_perf = (max_bus_bw / ideal_perf) * 100
        grade = min(max(normalized_perf, 0), 100)
        return grade

    def _extract_max_bus_bandwidth(self, stdout_path: str) -> float:
        """
        Extract the maximum bus bandwidth from the NcclTest output file.

        Args:
            stdout_path (str): Path to the stdout.txt file containing the NcclTest output.

        Returns:
            float: The maximum bus bandwidth value.
        """
        max_bus_bw = 0.0
        with open(stdout_path, "r") as file:
            for line in file:
                if line.strip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) > 10:  # Ensure it's a data line
                    out_of_place_bw = float(parts[7])
                    in_place_bw = float(parts[10])
                    max_bw = max(out_of_place_bw, in_place_bw)
                    max_bus_bw = max(max_bus_bw, max_bw)
        return max_bus_bw
