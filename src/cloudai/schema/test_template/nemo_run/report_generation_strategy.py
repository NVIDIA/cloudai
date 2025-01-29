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
import os
from pathlib import Path
from typing import Optional

import numpy as np

from cloudai import ReportGenerationStrategy


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    def can_handle_directory(self, directory_path: Path) -> bool:
        for _, __, files in os.walk(directory_path):
            for file in files:
                if file.startswith("stdout.txt"):
                    return True
        return False

    def generate_report(self, test_name: str, directory_path: Path, sol: Optional[float] = None) -> None:
        stdout_file = directory_path / "stdout.txt"
        if not stdout_file.exists():
            logging.error(f"{stdout_file} not found")
            return

        train_step_timings = []

        with open(stdout_file, "r") as f:
            for line in f:
                if "train_step_timing in s:" in line:
                    try:
                        timing = float(line.strip().split(":")[-1])
                        train_step_timings.append(timing)
                    except ValueError:
                        continue

        if not train_step_timings:
            logging.error(f"No train_step_timing found in {stdout_file}")
            return

        stats = {
            "avg": np.mean(train_step_timings),
            "median": np.median(train_step_timings),
            "min": np.min(train_step_timings),
            "max": np.max(train_step_timings),
        }

        summary_file = directory_path / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"{stats['avg']},{stats['median']},{stats['min']},{stats['max']}\n")
