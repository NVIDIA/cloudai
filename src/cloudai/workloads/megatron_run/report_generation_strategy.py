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
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

CHECKPOINT_REGEX = re.compile(r"(save|load)-checkpoint\s.*:\s\((\d+\.\d+),\s(\d+\.\d+)\)")

# Pattern to match lines like:
# [2026-01-16 07:32:39] iteration  6/100 | ... |
#   elapsed time per iteration (ms): 15639.0 | throughput per GPU (TFLOP/s/GPU): 494.6 | ...
ITERATION_REGEX = re.compile(
    r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


class CheckpointTimingReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from Checkpoint Timing test outputs."""

    def can_handle_directory(self) -> bool:
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                for line in file:
                    #     save-checkpoint ................................: (25976.43, 25976.63)
                    # load-checkpoint ................................: (17352.38, 17353.07)
                    if re.search(CHECKPOINT_REGEX, line):
                        return True
        return False

    def generate_report(self) -> None:
        save_timings, load_timings = [], []
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.is_file():
            with stdout_path.open("r") as file:
                for line in file:
                    m = re.search(CHECKPOINT_REGEX, line)
                    if not m:
                        continue

                    if m.group(1) == "save":
                        save_timings.append((float(m.group(2)), float(m.group(3))))
                    else:
                        load_timings.append((float(m.group(2)), float(m.group(3))))

        logging.info(f"Found {len(save_timings)} save timings and {len(load_timings)} load timings.")

        report_path = self.test_run.output_path / "report.csv"
        with report_path.open("w") as file:
            file.write("checkpoint_type,min,max\n")
            for checkpoint_type, timings in [("save", save_timings), ("load", load_timings)]:
                for t in timings:
                    file.write(f"{checkpoint_type},{t[0]},{t[1]}\n")


class MegatronRunReportGenerationStrategy(ReportGenerationStrategy):
    """Parse Megatron-Run stdout.txt for iteration time and GPU TFLOP/s per GPU."""

    metrics: ClassVar[list[str]] = ["default", "iteration-time", "tflops-per-gpu"]

    def get_log_file(self) -> Path | None:
        log = self.test_run.output_path / "stdout.txt"
        return log if log.is_file() else None

    @property
    def results_file(self) -> Path:
        return self.get_log_file() or (self.test_run.output_path / "stdout.txt")

    def can_handle_directory(self) -> bool:
        log_file = self.get_log_file()
        if not log_file:
            return False
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ITERATION_REGEX.search(line):
                    return True
        return False

    def _extract(self, log_path: Path) -> tuple[list[float], list[float]]:
        """Extract iteration times (ms) and GPU TFLOPS from the log file."""
        iter_times_ms: list[float] = []
        gpu_tflops: list[float] = []
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ITERATION_REGEX.search(line)
                if m:
                    try:
                        iter_times_ms.append(float(m.group(1)))
                        gpu_tflops.append(float(m.group(2)))
                    except (ValueError, TypeError):
                        logging.debug("Failed to parse iteration metrics line: %s", line.rstrip("\n"))

        # Keep only the last 10 iterations for statistics (to exclude warmup)
        if len(iter_times_ms) > 10:
            iter_times_ms = iter_times_ms[-10:]
            gpu_tflops = gpu_tflops[-10:]
        return iter_times_ms, gpu_tflops

    def _get_extracted_data(self) -> tuple[Path | None, list[float], list[float]]:
        log_file = self.get_log_file()
        if not log_file:
            return None, [], []
        iter_times_ms, gpu_tflops = self._extract(log_file)
        return log_file, iter_times_ms, gpu_tflops

    def generate_report(self) -> None:
        log_file, iter_times_ms, gpu_tflops = self._get_extracted_data()
        if not log_file:
            logging.error(
                "No stdout.txt file found in: %s",
                self.test_run.output_path,
            )
            return

        report_file = self.test_run.output_path / "megatron_run_report.csv"
        if not iter_times_ms:
            with report_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric_type", "avg", "median", "min", "max", "std"])
                writer.writerow(["error: No iteration timing lines were found.", "", "", "", "", ""])
            logging.warning("No iteration metrics found under %s (wrote %s)", self.test_run.output_path, report_file)
            return

        iter_avg = mean(iter_times_ms)
        iter_median = median(iter_times_ms)
        iter_min = min(iter_times_ms)
        iter_max = max(iter_times_ms)
        iter_std = pstdev(iter_times_ms) if len(iter_times_ms) > 1 else 0.0

        if gpu_tflops:
            tflops_avg = mean(gpu_tflops)
            tflops_median = median(gpu_tflops)
            tflops_min = min(gpu_tflops)
            tflops_max = max(gpu_tflops)
            tflops_std = pstdev(gpu_tflops) if len(gpu_tflops) > 1 else 0.0
        else:
            tflops_avg = tflops_median = tflops_min = tflops_max = tflops_std = 0.0

        with report_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_type", "avg", "median", "min", "max", "std"])
            writer.writerow(["iteration_time_ms", iter_avg, iter_median, iter_min, iter_max, iter_std])
            writer.writerow(["tflops_per_gpu", tflops_avg, tflops_median, tflops_min, tflops_max, tflops_std])

    def get_metric(self, metric: str) -> float:
        if metric not in {"default", "iteration-time", "tflops-per-gpu"}:
            return METRIC_ERROR
        log_file, iter_times_ms, gpu_tflops = self._get_extracted_data()
        if not log_file:
            logging.error(
                "No stdout.txt file found in: %s",
                self.test_run.output_path,
            )
            return METRIC_ERROR
        if not iter_times_ms:
            return METRIC_ERROR

        if metric in {"default", "iteration-time"}:
            return float(mean(iter_times_ms))
        if metric == "tflops-per-gpu":
            return float(mean(gpu_tflops)) if gpu_tflops else METRIC_ERROR
        return METRIC_ERROR
