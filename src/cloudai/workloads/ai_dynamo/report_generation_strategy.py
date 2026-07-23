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

import csv
import logging
import math
from pathlib import Path
from typing import ClassVar, Mapping

from cloudai.core import METRIC_ERROR, MetricValue, ReportGenerationStrategy
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.ai_dynamo.ai_dynamo import AIDynamoTestDefinition, parse_aiperf_accuracy
from cloudai.workloads.common.llm_serving import LLMServingBenchReport


class _AnyMetricSet(list):
    """List subclass that reports containment for any item; used to mark a reporter as handling all metrics."""

    def __contains__(self, item: object) -> bool:
        return True


class AIDynamoBenchReport(LLMServingBenchReport):
    """Normalized AI Dynamo benchmark results used by comparison reports."""

    output_throughput: float

    @property
    def throughput(self) -> float:
        return self.output_throughput


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    # Accepts any metric string — get_metric parses "benchmark:metric_name:column" dynamically.
    metrics: ClassVar[list[str]] = _AnyMetricSet()

    _TTFT_NAMES = ("Time To First Token (ms)", "Time to First Token (ms)")
    _TPOT_NAMES = ("Inter Token Latency (ms)",)
    _THROUGHPUT_NAMES = ("Output Token Throughput (tokens/sec)",)
    _REQUEST_COUNT_NAMES = ("Request Count (count)", "Request Count", "Request count")

    @staticmethod
    def _parse_metric_rows(csv_file: Path) -> dict[str, dict[str, str]]:
        """Parse both single-table GenAI-Perf and multi-table AIPerf CSV output."""
        metrics: dict[str, dict[str, str]] = {}
        header: list[str] | None = None
        with csv_file.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row or not any(cell.strip() for cell in row):
                    continue
                row = [cell.strip() for cell in row]
                if row[0] == "Metric":
                    header = row
                    continue
                if header is None:
                    continue
                metrics[row[0]] = {column: value for column, value in zip(header[1:], row[1:], strict=False) if value}
        return metrics

    @staticmethod
    def _find_value(
        metrics: Mapping[str, Mapping[str, str]], metric_names: tuple[str, ...], value_names: tuple[str, ...]
    ) -> float | None:
        for metric_name in metric_names:
            row = metrics.get(metric_name)
            if row is None:
                continue
            for value_name in value_names:
                value = row.get(value_name)
                if value is None:
                    continue
                try:
                    return float(value.replace(",", ""))
                except ValueError:
                    continue
        return None

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
        try:
            metrics = self._parse_metric_rows(csv_file)
        except (OSError, csv.Error) as e:
            logging.debug(f"Error parsing AI Dynamo benchmark output {csv_file}: {e}")
            return None

        ttft_mean = self._find_value(metrics, self._TTFT_NAMES, ("avg", "mean", "Value"))
        tpot_mean = self._find_value(metrics, self._TPOT_NAMES, ("avg", "mean", "Value"))
        throughput = self._find_value(metrics, self._THROUGHPUT_NAMES, ("avg", "Value"))
        if ttft_mean is None or tpot_mean is None or throughput is None:
            return None

        request_count = self._find_value(metrics, self._REQUEST_COUNT_NAMES, ("avg", "Value"))
        unavailable = math.nan
        return AIDynamoBenchReport(
            num_prompts=int(request_count or 0),
            completed=int(request_count or 0),
            mean_ttft_ms=ttft_mean,
            median_ttft_ms=self._find_value(metrics, self._TTFT_NAMES, ("p50", "median")) or unavailable,
            p99_ttft_ms=self._find_value(metrics, self._TTFT_NAMES, ("p99",)) or unavailable,
            mean_tpot_ms=tpot_mean,
            median_tpot_ms=self._find_value(metrics, self._TPOT_NAMES, ("p50", "median")) or unavailable,
            p99_tpot_ms=self._find_value(metrics, self._TPOT_NAMES, ("p99",)) or unavailable,
            output_throughput=throughput,
            max_concurrency=self._max_concurrency(),
        )

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
