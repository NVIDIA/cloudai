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

import json
from pathlib import Path

from cloudai import GradingStrategy


class DeepEPBenchmarkGradingStrategy(GradingStrategy):
    """
    Performance grading strategy for DeepEP Benchmark test templates.

    Evaluates the test's performance by comparing the global bandwidth or deepep time
    achieved during the test against the test's ideal performance metric.
    The grade is normalized and scaled between 0 and 100.
    """

    def grade(self, directory_path: Path, ideal_perf: float) -> float:
        """
        Grades the performance of a DeepEP Benchmark based on results.

        Args:
            directory_path (Path): Path to the directory containing the test's output.
            ideal_perf (float): The ideal performance value for comparison.

        Returns:
            float: The performance grade of the test, normalized and scaled between 0 and 100.
        """
        # Find all benchmark result directories
        results_dirs = list(directory_path.glob("results/benchmark_*_ranks_*"))
        
        if not results_dirs:
            return 0.0

        max_performance = 0.0

        for result_dir in results_dirs:
            results_json = result_dir / "results.json"
            if not results_json.exists():
                continue
            
            try:
                with open(results_json, 'r') as f:
                    results_data = json.load(f)
                
                # Extract the best global bandwidth from results
                for result in results_data:
                    # DeepEP results include global_bw, deepep_time, etc.
                    # We use global_bw as the primary performance metric
                    if 'global_bw' in result:
                        global_bw = float(result['global_bw'])
                        max_performance = max(max_performance, global_bw)
                    elif 'deepep_time' in result:
                        # If only time is available, use inverse (lower time = better)
                        # For grading, we'd need ideal_perf to be in time units
                        deepep_time = float(result['deepep_time'])
                        if deepep_time > 0:
                            # Convert time to throughput-like metric
                            perf = 1000.0 / deepep_time  # arbitrary scaling
                            max_performance = max(max_performance, perf)
            
            except Exception as e:
                # Skip invalid result files
                continue

        if max_performance <= 0 or ideal_perf <= 0:
            return 0.0

        # Calculate normalized grade
        normalized_perf = (max_performance / ideal_perf) * 100
        grade = min(max(normalized_perf, 0), 100)
        return grade

