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
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from cloudai.core import ReportGenerationStrategy
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool


class DeepEPBenchmarkReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from DeepEP benchmark outputs.
    
    Parses results.json and generates CSV summaries.
    """

    def can_handle_directory(self, directory_path: Path) -> bool:
        """
        Check if this directory contains DeepEP benchmark results.
        
        Args:
            directory_path (Path): Path to the directory to check.
        
        Returns:
            bool: True if directory contains DeepEP results.
        """
        # Check for results subdirectories created by DeepEP
        results_pattern = directory_path / "results" / "benchmark_*_ranks_*"
        matching_dirs = list(directory_path.glob("results/benchmark_*_ranks_*"))
        
        if matching_dirs:
            # Check if any of them has results.json
            for result_dir in matching_dirs:
                if (result_dir / "results.json").exists():
                    return True
        
        return False

    def generate_report(self, test_name: str, directory_path: Path, sol: Optional[float] = None) -> None:
        """
        Generate a report from DeepEP benchmark results.
        
        Args:
            test_name (str): Name of the test.
            directory_path (Path): Directory containing the results.
            sol (Optional[float]): Speed-of-light performance for reference.
        """
        # Find all benchmark result directories
        results_dirs = list(directory_path.glob("results/benchmark_*_ranks_*"))
        
        if not results_dirs:
            return
        
        all_results = []
        
        for result_dir in results_dirs:
            results_json = result_dir / "results.json"
            if not results_json.exists():
                continue
            
            try:
                with open(results_json, 'r') as f:
                    results_data = json.load(f)
                
                # Extract metadata from directory name
                dir_name = result_dir.name
                # Format: benchmark_{num_ranks}_ranks_{timestamp}_{mode}
                match = re.match(r'benchmark_(\d+)_ranks_(.+)_(\w+)', dir_name)
                if match:
                    num_ranks = int(match.group(1))
                    timestamp = match.group(2)
                    mode = match.group(3)
                else:
                    num_ranks = 0
                    timestamp = "unknown"
                    mode = "unknown"
                
                # Process results
                for result in results_data:
                    result['num_ranks'] = num_ranks
                    result['timestamp'] = timestamp
                    result['mode'] = mode
                    result['result_dir'] = str(result_dir)
                    all_results.append(result)
            
            except Exception as e:
                print(f"Error parsing {results_json}: {e}")
                continue
        
        if all_results:
            # Create DataFrame
            df = pd.DataFrame(all_results)
            
            # Reorder columns for better readability
            column_order = ['mode', 'num_ranks', 'num_tokens', 'hidden', 
                          'deepep_time', 'global_bw', 'simple_rdma_bw', 'simple_nvl_bw',
                          'timestamp', 'result_dir']
            
            # Only include columns that exist
            column_order = [col for col in column_order if col in df.columns]
            df = df[column_order]
            
            # Generate CSV report
            self._generate_csv_report(df, directory_path, test_name)

    def _generate_csv_report(self, df: pd.DataFrame, directory_path: Path, test_name: str) -> None:
        """
        Generate a CSV report from the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the benchmark results.
            directory_path (Path): Output directory path for saving the CSV report.
            test_name (str): Name of the test.
        """
        csv_report_tool = CSVReportTool(directory_path)
        csv_report_tool.set_dataframe(df)
        csv_report_tool.finalize_report(Path(f"cloudai_{test_name}_report.csv"))

