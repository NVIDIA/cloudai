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

from cloudai import ReportGenerationStrategy

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
