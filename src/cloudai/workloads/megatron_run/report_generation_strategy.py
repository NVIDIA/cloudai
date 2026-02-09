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
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

CHECKPOINT_REGEX = re.compile(r"(save|load)-checkpoint\s.*:\s\((\d+\.\d+),\s(\d+\.\d+)\)")


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
    """Parse Megatron-LM training logs for step time and GPU TFLOP/s per GPU."""

    metrics: ClassVar[list[str]] = ["default", "step-time", "tflops-per-gpu"]

    ITERATION_LINE_RE = re.compile(
        r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"
        r".*?"
        r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    def get_log_file(self) -> Path | None:
        """Find the stdout log file containing Megatron training output."""
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.is_file():
            return stdout_path
        return None

    @property
    def results_file(self) -> Path:
        return self.get_log_file() or (self.test_run.output_path / "stdout.txt")

    def can_handle_directory(self) -> bool:
        """Check if directory contains Megatron training logs with iteration metrics."""
        log_file = self.get_log_file()
        if not log_file:
            return False

        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if self.ITERATION_LINE_RE.search(line):
                    return True
        return False

    def _extract(self, log_path: Path) -> tuple[list[float], list[float]]:
        """Extract step times (in seconds) and GPU TFLOP/s from log file."""
        step_times_s: list[float] = []
        gpu_tflops: list[float] = []

        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = self.ITERATION_LINE_RE.search(line)
                if m:
                    try:
                        elapsed_ms = float(m.group(1))
                        step_times_s.append(elapsed_ms / 1000.0)
                        gpu_tflops.append(float(m.group(2)))
                    except (ValueError, TypeError):
                        logging.debug("Failed to parse iteration metrics line: %s", line.rstrip("\n"))

        if len(step_times_s) > 10:
            step_times_s = step_times_s[-10:]
            gpu_tflops = gpu_tflops[-10:]

        return step_times_s, gpu_tflops

    def _get_extracted_data(self) -> tuple[Path | None, list[float], list[float]]:
        """Get log file and extracted metrics data."""
        log_file = self.get_log_file()
        if not log_file:
            return None, [], []
        step_times_s, gpu_tflops = self._extract(log_file)
        return log_file, step_times_s, gpu_tflops

    def generate_report(self) -> None:
        """Generate a summary report with step time and TFLOP/s statistics."""
        log_file, step_times_s, gpu_tflops = self._get_extracted_data()
        if not log_file:
            logging.error(
                "No Megatron training log file found in: %s",
                self.test_run.output_path,
            )
            return

        summary_file = self.test_run.output_path / "report.txt"
        if not step_times_s:
            with summary_file.open("w") as f:
                f.write("MegatronRun report\n")
                f.write("No iteration metrics found in log.\n\n")
                f.write("Expected log format:\n")
                f.write("  elapsed time per iteration (ms): X.X | throughput per GPU (TFLOP/s/GPU): X.X\n\n")
                f.write("Searched file:\n")
                f.write(f"  - {log_file}\n")
            logging.warning("No iteration metrics found under %s (wrote %s)", self.test_run.output_path, summary_file)
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
            f.write(f"Source log: {log_file}\n\n")
            f.write("Step Time (s)\n")
            f.write(f"  avg: {step_stats['avg']:.4f}\n")
            f.write(f"  median: {step_stats['median']:.4f}\n")
            f.write(f"  min: {step_stats['min']:.4f}\n")
            f.write(f"  max: {step_stats['max']:.4f}\n")
            f.write(f"  std: {step_stats['std']:.4f}\n")
            f.write("\n")
            f.write("TFLOP/s per GPU\n")
            f.write(f"  avg: {tflops_stats['avg']:.2f}\n")
            f.write(f"  median: {tflops_stats['median']:.2f}\n")
            f.write(f"  min: {tflops_stats['min']:.2f}\n")
            f.write(f"  max: {tflops_stats['max']:.2f}\n")
            f.write(f"  std: {tflops_stats['std']:.2f}\n")

        logging.info("Generated MegatronRun report: %s", summary_file)

    def get_metric(self, metric: str) -> float:
        """Get a specific metric value for DSE/optimization."""
        if metric not in {"default", "step-time", "tflops-per-gpu"}:
            return METRIC_ERROR

        log_file, step_times_s, gpu_tflops = self._get_extracted_data()
        if not log_file:
            logging.error(
                "No Megatron training log file found in: %s",
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
