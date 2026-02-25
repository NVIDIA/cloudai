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

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.util.lazy_imports import lazy


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    def extract_metric_from_csv(self, csv_file: Path, metric_name: str, metric_type: str) -> float:
        df = lazy.pd.read_csv(csv_file)

        if "Metric" not in df.columns or metric_type not in df.columns:
            logging.info(f"Metric type: {metric_type} not in CSV file: {df.columns}")
            return METRIC_ERROR

        if metric_name not in df["Metric"].values:
            logging.info(f"Metric name: {metric_name} not in CSV file: {df['Metric'].values}")
            return METRIC_ERROR

        series = df.loc[df["Metric"] == metric_name, metric_type]
        if series.empty:
            return METRIC_ERROR
        return float(series.iloc[0])

    def get_metric(self, metric: str) -> float:
        logging.info(f"Getting metric: {metric}")
        benchmark_name = "genai_perf"
        metric_name = metric
        metric_type = "avg"

        if ":" in metric:
            parts = metric.split(":", maxsplit=2)
            if len(parts) != 3:
                logging.warning(f"Invalid metric format: {metric}. Expected 'benchmark:metric_name:metric_type'")
                return METRIC_ERROR
            benchmark_name, metric_name, metric_type = parts

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
