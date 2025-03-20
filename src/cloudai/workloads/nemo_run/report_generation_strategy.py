# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR


@cache
def extract_timings(stdout_file: Path) -> tuple[list[float], list[float]]:
    train_step_timings: list[float] = []
    step_timings: list[float] = []

    with open(stdout_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "train_step_timing in s:" in line:
                try:
                    timing = float(line.split("train_step_timing in s:")[1].strip().split()[0])
                    train_step_timings.append(timing)
                    if "global_step:" in line:
                        global_step = int(line.split("global_step:")[1].split("|")[0].strip())
                        if 80 <= global_step <= 100:
                            step_timings.append(timing)
                except (ValueError, IndexError):
                    continue

    if not train_step_timings:
        logging.error(f"No train_step_timing found in {stdout_file}")
        return [], []

    if len(step_timings) < 20:
        step_timings = train_step_timings[1:]

    return train_step_timings, step_timings


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt"):
                    return True
        return False

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def generate_report(self) -> None:
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return

        train_step_timings, step_timings = extract_timings(self.results_file)

        stats = {
            "avg": np.mean(step_timings),
            "median": np.median(step_timings),
            "min": np.min(step_timings),
            "max": np.max(step_timings),
        }

        summary_file = self.test_run.output_path / "report.txt"
        with open(summary_file, "w") as f:
            f.write("Average: {avg}\n".format(avg=stats["avg"]))
            f.write("Median: {median}\n".format(median=stats["median"]))
            f.write("Min: {min}\n".format(min=stats["min"]))
            f.write("Max: {max}\n".format(max=stats["max"]))

    def get_metric(self, metric: str) -> float:
        if metric == "default" or metric == "step-time":
            _, step_timings = extract_timings(self.results_file)
            if not step_timings:
                return METRIC_ERROR
            return float(np.mean(step_timings))

        return METRIC_ERROR
