# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from cloudai.core import ReportGenerationStrategy
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class DeepEPReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from DeepEP benchmark outputs."""

    def can_handle_directory(self) -> bool:
        """
        Check if this directory contains DeepEP benchmark results.

        Returns:
            bool: True if directory contains DeepEP results.
        """
        # Check for results subdirectories created by DeepEP
        directory_path = self.test_run.output_path
        matching_dirs = list(directory_path.glob("results/benchmark_*_ranks_*"))

        if matching_dirs:
            # Check if any of them has results.json
            for result_dir in matching_dirs:
                if (result_dir / "results.json").exists():
                    return True

        return False

    def generate_report(self) -> None:
        """Generate a report from DeepEP benchmark results."""
        directory_path = self.test_run.output_path
        test_name = self.test_run.test.name

        results_dirs = list(directory_path.glob("results/benchmark_*_ranks_*"))

        if not results_dirs:
            return

        all_results = []

        for result_dir in results_dirs:
            results_json = result_dir / "results.json"
            if not results_json.exists():
                continue

            try:
                with open(results_json, "r") as f:
                    results_data = json.load(f)
            except Exception as e:
                logging.debug(f"Error parsing {results_json}: {e}")
                continue

            match = re.match(r"benchmark_(\d+)_ranks_(.+?)_(low_latency|standard)", result_dir.name)
            num_ranks, timestamp, mode = 0, "unknown", "unknown"
            if match:
                num_ranks = int(match.group(1))
                timestamp = match.group(2)
                mode = match.group(3)

            for result in results_data:
                result["num_ranks"] = num_ranks
                result["timestamp"] = timestamp
                result["mode"] = mode
                result["result_dir"] = str(result_dir)
                all_results.append(result)

        if all_results:
            df = lazy.pd.DataFrame(all_results)

            column_order = [
                "mode",
                "num_ranks",
                "num_tokens",
                "hidden",
                "deepep_time",
                "global_bw",
                "simple_rdma_bw",
                "simple_nvl_bw",
                "timestamp",
                "result_dir",
            ]

            column_order = [col for col in column_order if col in df.columns]
            df = df[column_order]

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
