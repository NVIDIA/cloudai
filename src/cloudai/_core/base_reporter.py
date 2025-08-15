# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import toml

from .system import System
from .test_scenario import TestRun, TestScenario

if TYPE_CHECKING:
    from ..models.scenario import ReportConfig


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
        for _tr in self.test_scenario.test_runs:
            tr = copy.deepcopy(_tr)
            tr_root = self.results_root / tr.name
            iters = list(subdir for subdir in tr_root.glob("*") if subdir.is_dir())
            for iter in sorted(iters, key=lambda x: int(x.name)):
                if tr.is_dse_job:
                    steps = list(subdir for subdir in iter.glob("*") if subdir.is_dir())
                    for step in sorted(steps, key=lambda x: int(x.name)):
                        tr.current_iteration = int(iter.name)
                        tr.step = int(step.name)
                        tr.output_path = tr_root / f"{tr.current_iteration}" / f"{tr.step}"
                        tr_file = tr.output_path / "test-run.toml"
                        if tr_file.exists():
                            tr_file = toml.load(tr_file)
                            tr.test.test_definition = tr.test.test_definition.model_validate(tr_file["test_definition"])
                        self.trs.append(copy.deepcopy(tr))
                else:
                    tr.current_iteration = int(iter.name)
                    tr.step = 0
                    tr.output_path = tr_root / f"{tr.current_iteration}"
                    self.trs.append(copy.deepcopy(tr))

        logging.debug(f"Loaded {len(self.trs)} test runs for {self.test_scenario.name} in {self.results_root}")
        for tr in self.trs:
            logging.debug(f"Test run: {tr.name} {tr.output_path}")

    @abstractmethod
    def generate(self) -> None:
        """Generate the report."""
        ...
