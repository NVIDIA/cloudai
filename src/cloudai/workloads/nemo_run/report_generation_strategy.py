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
from typing import ClassVar, List

import numpy as np
import pandas as pd

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool


@cache
def extract_timings(stdout_file: Path) -> list[float]:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return []

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
        logging.debug(f"No train_step_timing found in {stdout_file}")
        return []

    if len(step_timings) < 20:
        step_timings = train_step_timings[1:]

    return step_timings


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt") and extract_timings(self.test_run.output_path / file):
                    return True
        return False

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def generate_report(self) -> None:
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return

        step_timings = extract_timings(self.results_file)
        if not step_timings:
            logging.error(f"No valid step timings found in {self.results_file}. Report generation aborted.")
            return

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

        self.generate_bokeh_report(step_timings)

    def get_metric(self, metric: str) -> float:
        logging.debug(f"Getting metric {metric} from {self.results_file.absolute()}")
        step_timings = extract_timings(self.results_file)
        if not step_timings:
            return METRIC_ERROR

        if metric not in {"default", "step-time"}:
            return METRIC_ERROR

        return float(np.mean(step_timings))

    def generate_bokeh_report(self, step_timings: List[float]) -> None:
        if not step_timings:
            return

        df = pd.DataFrame({"Step": range(1, len(step_timings) + 1), "train_step_timing in s": step_timings})

        report_tool = BokehReportTool(self.test_run.output_path)
        report_tool.add_linear_xy_line_plot(
            title="Train Step Timing over Steps",
            x_column="Step",
            y_column="train_step_timing in s",
            x_axis_label="Step",
            df=df,
            sol=self.test_run.sol,
            color="black",
        )
        report_tool.finalize_report(Path("cloudai_nemorun_bokeh_report.html"))
