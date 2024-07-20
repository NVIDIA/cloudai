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

from cloudai import GradingStrategy


class ChakraReplayGradingStrategy(GradingStrategy):
    """Performance grading strategy for ChakraReplay test templates on Slurm systems."""

    def grade(self, directory_path: str, ideal_perf: float) -> float:
        """
        Grades the performance of a test.

        Args:
            directory_path (str): Path to the directory containing the
                                  test's output.
            ideal_perf (float): The ideal performance value for comparison.

        Returns:
            float: Calculated grade based on the performance.
        """
        return 100.0
