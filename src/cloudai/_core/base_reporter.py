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

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .system import System

if TYPE_CHECKING:
    from ..models.scenario import ReportConfig, TestScenario
    from .test_run import TestRun


def case_name(tr: TestRun) -> str:
    """Generate a display name for a test run including iteration and step if applicable."""
    name = tr.name
    if tr.current_iteration > 0:
        name = f"{name} iter={tr.current_iteration}"
    if tr.step > 0:
        name = f"{name} step={tr.step}"
    return name


class Reporter(ABC):
    """Abstract base class for all reporters."""

    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path, config: ReportConfig) -> None:
        self.system = system
        self.test_scenario = test_scenario
        self.results_root = results_root
        self.trs: list[TestRun] = []
        self.config = config

    def load_test_runs(self):
        """Load test runs from the results directory."""
        from .test_run_io import load_test_run

        for template in self.test_scenario.test_runs:
            tr_root = self.results_root / template.name
            iterations = sorted((path for path in tr_root.glob("*") if path.is_dir()), key=lambda path: int(path.name))
            for iteration in iterations:
                paths = [iteration]
                if template.is_dse_job:
                    paths = sorted(
                        (path for path in iteration.iterdir() if path.is_dir()), key=lambda path: int(path.name)
                    )
                for output_path in paths:
                    snapshot = output_path / "test-run.toml"
                    run = load_test_run(snapshot, template.test) if snapshot.is_file() else copy.deepcopy(template)
                    run.current_iteration = int(iteration.name)
                    run.step = int(output_path.name) if template.is_dse_job else 0
                    run.output_path = output_path
                    self.trs.append(run)

        logging.debug(f"Loaded {len(self.trs)} test runs for {self.test_scenario.name} in {self.results_root}")
        for tr in self.trs:
            logging.debug(f"Test run: {tr.name} {tr.output_path}")

    @abstractmethod
    def generate(self) -> None:
        """Generate the report."""
        ...
