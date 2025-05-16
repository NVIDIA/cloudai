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

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .system import System
from .test_scenario import TestRun, TestScenario


class Reporter(ABC):
    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path) -> None:
        self.system = system
        self.test_scenario = test_scenario
        self.results_root = results_root
        self.trs: list[TestRun] = []

    def load_test_runs(self):
        for _tr in self.test_scenario.test_runs:
            tr = copy.deepcopy(_tr)
            tr_root = self.results_root / tr.name
            iters = list(subdir for subdir in tr_root.glob("*") if subdir.is_dir())
            for iter in sorted(iters, key=lambda x: int(x.name)):
                if tr.test.test_definition.is_dse_job:
                    steps = list(subdir for subdir in iter.glob("*") if subdir.is_dir())
                    for step in sorted(steps, key=lambda x: int(x.name)):
                        tr.current_iteration = int(iter.name)
                        tr.step = int(step.name)
                        tr.output_path = tr_root / f"{tr.current_iteration}" / f"{tr.step}"
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
    def generate(self) -> None: ...
