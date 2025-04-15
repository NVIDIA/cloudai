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
from pathlib import Path
from typing import ClassVar, Dict, List

import numpy as np

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR

from .report_utils import parse_step_timings


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[List[str]] = ["default", "step-time"]

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt") and parse_step_timings(self.test_run.output_path / file):
                    return True
        return False

    def generate_report(self) -> None:
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return
        train_step_timings: List[float] = parse_step_timings(self.results_file)
        if not train_step_timings:
            logging.error(f"No valid step step_timings found in {self.results_file}. Report generation aborted.")
            return
        self._write_summary_file(self._compute_statistics(train_step_timings))

    def _compute_statistics(self, step_timings: List[float]) -> Dict[str, float]:
        return {
            "avg": float(np.mean(step_timings)),
            "median": float(np.median(step_timings)),
            "min": float(np.min(step_timings)),
            "max": float(np.max(step_timings)),
        }

    def _write_summary_file(self, stats: Dict[str, float]) -> None:
        summary_file: Path = self.test_run.output_path / "report.txt"
        with open(summary_file, "w") as f:
            f.write("Average: {avg}\n".format(avg=stats["avg"]))
            f.write("Median: {median}\n".format(median=stats["median"]))
            f.write("Min: {min}\n".format(min=stats["min"]))
            f.write("Max: {max}\n".format(max=stats["max"]))

    def get_metric(self, metric: str) -> float:
        logging.debug(f"Getting metric {metric} from {self.results_file.absolute()}")
        step_timings = parse_step_timings(self.results_file)
        if not step_timings:
            return METRIC_ERROR

        if metric not in {"default", "step-time"}:
            return METRIC_ERROR

        return float(np.mean(step_timings))
