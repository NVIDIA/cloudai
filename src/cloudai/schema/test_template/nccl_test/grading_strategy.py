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

from pathlib import Path

from cloudai import GradingStrategy

from .output_reader_mixin import NcclTestOutputReaderMixin


class NcclTestGradingStrategy(NcclTestOutputReaderMixin, GradingStrategy):
    """
    Base grading strategy for NCCL tests.

    Evaluates the test's performance by comparing the maximum bus bandwidth achieved during the test against the test's
    ideal performance metric. The grade is normalized and scaled between 0 and 100.
    """

    def grade(self, directory_path: Path, ideal_perf: float) -> float:
        """
        Grades the performance of an NCCL test based on the maximum bus bandwidth.

        Args:
            directory_path (Path): Path to the directory containing the test's output.
            ideal_perf (float): The ideal performance value for comparison.

        Returns:
            float: The performance grade of the test, normalized and scaled between 0 and 100.
        """
        content = self._get_stdout_content(directory_path)
        if content is None:
            return 0.0

        max_bus_bw = self._extract_max_bus_bandwidth(content)
        if max_bus_bw is None or ideal_perf <= 0:
            return 0.0

        normalized_perf = (max_bus_bw / ideal_perf) * 100
        return min(max(normalized_perf, 0), 100)

    def _extract_max_bus_bandwidth(self, content: str) -> float:
        """
        Extract the maximum bus bandwidth from the NCCL test output content.

        Args:
            content (str): The content of the stdout file.

        Returns:
            float: The maximum bus bandwidth value.
        """
        max_bus_bw = 0.0
        for line in content.splitlines():
            if line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) > 10:  # Ensure it's a data line
                out_of_place_bw = float(parts[7])
                in_place_bw = float(parts[10])
                max_bw = max(out_of_place_bw, in_place_bw)
                max_bus_bw = max(max_bus_bw, max_bw)
        return max_bus_bw
