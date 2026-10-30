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
from typing import ClassVar

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.ai_dynamo.ai_dynamo import (
    AIDynamoBenchReport,
    AIDynamoTestDefinition,
    parse_aiperf_accuracy,
)


class _AnyMetricSet(list):
    """List subclass that reports containment for any item; used to mark a reporter as handling all metrics."""

    def __contains__(self, item: object) -> bool:
        return True


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    # Accepts any metric string — get_metric parses "benchmark:metric_name:column" dynamically.
    metrics: ClassVar[list[str]] = _AnyMetricSet()

    def _benchmark_csv(self) -> Path | None:
        tdef = self.test_run.test
        if not isinstance(tdef, AIDynamoTestDefinition):
            return None
        for workload in tdef.cmd_args.workloads_list:
            workload_name = Path(workload).stem
            workload_config = getattr(tdef.cmd_args, workload_name, None)
            report_name = getattr(workload_config, "report_name", f"{workload_name}_report.csv")
            csv_file = self.test_run.output_path / report_name
            if csv_file.is_file() and csv_file.stat().st_size:
                return csv_file
        return None

    def _max_concurrency(self) -> int:
        tdef = self.test_run.test
        if not isinstance(tdef, AIDynamoTestDefinition):
            return 1

        workload_name = Path(tdef.cmd_args.workloads_list[0]).stem
        workload = getattr(tdef.cmd_args, workload_name)
        args = workload.args.model_dump()
        if workload_name == "aiperf" and tdef.cmd_args.aiperf_phases:
            args.update(tdef.cmd_args.aiperf_phases[-1].args.model_dump())
        try:
            return int(args.get("concurrency", 1))
        except (TypeError, ValueError):
            return 1

    def parse_results(self) -> AIDynamoBenchReport | None:
        """Normalize the configured benchmark CSV into the shared LLM serving result shape."""
        csv_file = self._benchmark_csv()
        if csv_file is None:
            return None
        return AIDynamoTestDefinition.parse_benchmark_csv(csv_file, self._max_concurrency())

    def used_gpus_count(self) -> int:
        """Match the GPU accounting used when AI Dynamo produces its result CSV."""
        return max(1, self.test_run.nnodes * (getattr(self.system, "gpus_per_node", 1) or 1))

    def parse_accuracy(self) -> float | None:
        return parse_aiperf_accuracy(self.test_run.output_path)

    def extract_metric_from_csv(self, csv_file: Path, metric_name: str, metric_type: str) -> MetricValue:
        df = lazy.pd.read_csv(csv_file, on_bad_lines="skip")

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

    def get_metric(self, metric: str) -> MetricValue:
        logging.info(f"Getting metric: {metric}")

        if metric.lower() == "accuracy":
            tdef = self.test_run.test
            if not isinstance(tdef, AIDynamoTestDefinition):
                return METRIC_ERROR
            if tdef.cmd_args.aiperf_accuracy is None:
                return METRIC_ERROR
            accuracy = parse_aiperf_accuracy(self.test_run.output_path)
            return accuracy if accuracy is not None else METRIC_ERROR

        metric_name = metric
        metric_type = "avg"

        if ":" in metric:
            parts = metric.split(":", maxsplit=2)
            if len(parts) != 3:
                logging.warning(f"Invalid metric format: {metric}. Expected 'benchmark:metric_name:metric_type'")
                return METRIC_ERROR
            benchmark_name, metric_name, metric_type = parts
        else:
            # Derive from the configured workload script (e.g. "aiperf.sh" → "aiperf").
            workloads_list = getattr(getattr(self.test_run.test, "cmd_args", None), "workloads_list", None)
            benchmark_name = Path(workloads_list[0]).stem if workloads_list else "aiperf"

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
