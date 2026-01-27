# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

if TYPE_CHECKING:
    pass

CSV_FILES_PATTERN = "profile*_genai_perf.csv"
JSON_FILES_PATTERN = "profile*_genai_perf.json"


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    def extract_metric_from_csv(self, csv_file: Path, metric_name: str, metric_type: str) -> float:
        df = pd.read_csv(csv_file)
        if metric_type not in df.columns:
            logging.info(f"Metric type: {metric_type} not in CSV file: {df.columns}")
            return METRIC_ERROR

        if metric_name not in df["Metric"].values:
            logging.info(f"Metric name: {metric_name} not in CSV file: {df['Metric'].values}")
            return METRIC_ERROR

        return float(df[df["Metric"] == metric_name][metric_type].values[0])

    def get_metric(self, metric: str) -> float:
        logging.info(f"Getting metric: {metric}")
        benchmark_name = "genai_perf"
        metric_name = metric
        metric_type = "avg"

        if ":" in metric:
            benchmark_name, metric_name, metric_type = metric.split(":")

        source_csv = self.test_run.output_path / f"{benchmark_name}_report.csv"
        logging.info(f"CSV file: {source_csv}")
        if not source_csv.exists() or source_csv.stat().st_size == 0:
            logging.info(f"CSV file: {source_csv} does not exist or is empty")
            return METRIC_ERROR

        return self.extract_metric_from_csv(source_csv, metric_name, metric_type)

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        pass
