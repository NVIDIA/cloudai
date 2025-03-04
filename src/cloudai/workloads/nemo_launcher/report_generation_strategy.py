# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from cloudai import ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool


class NeMoLauncherReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMo launcher directories."""

    def can_handle_directory(self) -> bool:
        run_dir = self.test_run.output_path / "run"
        return (
            run_dir.exists()
            and run_dir.is_dir()
            and any(file.name.startswith("log") and file.name.endswith(".out") for file in run_dir.iterdir())
        )

    def extract_train_step_timings(self) -> List[float]:
        run_dir = self.test_run.output_path / "run"
        log_file = next((file for file in run_dir.iterdir() if re.match(r"log-.*\.out", file.name)), None)

        if not log_file:
            logging.error("No valid log file found in the run directory")
            return []

        with open(log_file, "r") as f:
            for line in f:
                match = re.search(r"train_step_timing in s: \[([\d.,\s]+)\]", line)
                if match:
                    try:
                        step_timings = [float(val) for val in match.group(1).split(",")]
                        return self._filter_step_timings(step_timings)
                    except ValueError:
                        logging.error(f"Error parsing train step timings in {log_file}")
                        return []

        logging.error(f"No train step timings found in {log_file}")
        return []

    def _filter_step_timings(self, step_timings: List[float]) -> List[float]:
        return step_timings[-20:] if len(step_timings) > 100 else step_timings

    def generate_statistics_report(self, train_step_timings: List[float]) -> None:
        if not train_step_timings:
            return

        stats = {
            "avg": np.mean(train_step_timings),
            "median": np.median(train_step_timings),
            "min": np.min(train_step_timings),
            "max": np.max(train_step_timings),
        }

        summary_file = self.test_run.output_path / "train_step_timing_report.txt"
        with open(summary_file, "w") as f:
            f.writelines([f"{key.capitalize()}: {value:.4f}\n" for key, value in stats.items()])

    def generate_bokeh_report(self, train_step_timings: List[float]) -> None:
        if not train_step_timings:
            return

        df = pd.DataFrame({"Step": range(1, len(train_step_timings) + 1), "train_step_timing in s": train_step_timings})

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
        report_tool.finalize_report(Path("cloudai_nemo_launcher_bokeh_report.html"))

    def generate_report(self) -> None:
        train_step_timings = self.extract_train_step_timings()
        self.generate_statistics_report(train_step_timings)
        self.generate_bokeh_report(train_step_timings)
