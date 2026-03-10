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
import json
import logging
from pathlib import Path
from statistics import mean, median, pstdev
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

from .megatron_bridge import GOLDEN_VALUES_FILENAME


class MegatronBridgeReportGenerationStrategy(ReportGenerationStrategy):
    """Parse Megatron-Bridge JSON metrics file for step time and GPU TFLOP/s per GPU."""

    metrics: ClassVar[list[str]] = ["default", "step-time", "tflops-per-gpu"]

    @property
    def metrics_file(self) -> Path | None:
        for path in Path(self.test_run.output_path).rglob(GOLDEN_VALUES_FILENAME):
            return path
        return None

    def can_handle_directory(self) -> bool:
        return self.metrics_file is not None

    def _extract(self, metrics_file: Path) -> tuple[list[float], list[float]]:
        try:
            data: dict[str, dict[str, float]] = json.loads(metrics_file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted JSON metrics file at {metrics_file}") from e

        data = {k: v for k, v in data.items() if k.isdigit()}
        steps = sorted(list(map(int, data.keys())))

        try:
            step_times_s: list[float] = [data[str(step)]["elapsed time per iteration (ms)"] / 1000.0 for step in steps]
            # despite the metric is named GPU utilization, it's actually TFLOP/s per GPU
            gpu_tflops: list[float] = [data[str(step)]["GPU utilization"] for step in steps]
        except (KeyError, AttributeError, TypeError) as e:
            raise ValueError(f"Cannot parse JSON metrics file at {metrics_file} with {e}") from e

        if len(step_times_s) > 10:
            step_times_s = step_times_s[-10:]
            gpu_tflops = gpu_tflops[-10:]

        return step_times_s, gpu_tflops

    def _get_extracted_data(self) -> tuple[Path | None, list[float], list[float]]:
        metrics_file = self.metrics_file
        if not metrics_file:
            return None, [], []
        step_times_s, gpu_tflops = self._extract(metrics_file)
        return metrics_file, step_times_s, gpu_tflops

    def generate_report(self) -> None:
        metrics_file, step_times_s, gpu_tflops = self._get_extracted_data()
        if not metrics_file:
            logging.error(
                "No Megatron-Bridge launcher log file found in: %s",
                self.test_run.output_path,
            )
            return

        summary_file = self.test_run.output_path / "report.txt"
        if not step_times_s:
            with summary_file.open("w") as f:
                f.write("MegatronBridge report\n")
                f.write("No metrics were found.\n\n")
                f.write("Searched file:\n")
                f.write(f"  - {metrics_file}\n")
            logging.warning("No step metrics found under %s (wrote %s)", self.test_run.output_path, summary_file)
            return

        step_stats = {
            "avg": mean(step_times_s),
            "median": median(step_times_s),
            "min": min(step_times_s),
            "max": max(step_times_s),
            "std": pstdev(step_times_s) if len(step_times_s) > 1 else 0.0,
        }
        if gpu_tflops:
            tflops_stats = {
                "avg": mean(gpu_tflops),
                "median": median(gpu_tflops),
                "min": min(gpu_tflops),
                "max": max(gpu_tflops),
                "std": pstdev(gpu_tflops) if len(gpu_tflops) > 1 else 0.0,
            }
        else:
            tflops_stats = {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        with summary_file.open("w") as f:
            f.write(f"Source log: {metrics_file}\n\n")
            f.write("Step Time (s)\n")
            f.write(f"  avg: {step_stats['avg']}\n")
            f.write(f"  median: {step_stats['median']}\n")
            f.write(f"  min: {step_stats['min']}\n")
            f.write(f"  max: {step_stats['max']}\n")
            f.write(f"  std: {step_stats['std']}\n")
            f.write("\n")
            f.write("TFLOP/s per GPU\n")
            f.write(f"  avg: {tflops_stats['avg']}\n")
            f.write(f"  median: {tflops_stats['median']}\n")
            f.write(f"  min: {tflops_stats['min']}\n")
            f.write(f"  max: {tflops_stats['max']}\n")
            f.write(f"  std: {tflops_stats['std']}\n")

    def get_metric(self, metric: str) -> float:
        if metric not in {"default", "step-time", "tflops-per-gpu"}:
            return METRIC_ERROR
        metrics_file, step_times_s, gpu_tflops = self._get_extracted_data()
        if not metrics_file:
            logging.error(
                "No Megatron-Bridge launcher metrics file found in: %s",
                self.test_run.output_path,
            )
            return METRIC_ERROR
        if not step_times_s:
            return METRIC_ERROR

        if metric in {"default", "step-time"}:
            return float(mean(step_times_s))
        if metric == "tflops-per-gpu":
            return float(mean(gpu_tflops)) if gpu_tflops else METRIC_ERROR
        return METRIC_ERROR
