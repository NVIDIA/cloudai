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

"""Data abstraction layer for CloudAI UI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import toml

from cloudai._core.test_scenario import TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm.slurm_system import SlurmSystem


@dataclass
class TestScenarioInfo:
    """Information about a test scenario run."""

    id: str
    name: str
    timestamp: datetime
    test_runs: list[TestRun]
    error: str | None = None


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_scenarios(self) -> list[TestScenarioInfo]:
        """Get list of all test scenarios."""
        pass


class LocalFileDataProvider(DataProvider):
    """Data provider that reads from local filesystem."""

    def __init__(self, results_root: Path):
        self.results_root = Path(results_root)

    def get_scenarios(self) -> list[TestScenarioInfo]:
        """Get list of all test scenarios from the results directory."""
        scenarios: list[TestScenarioInfo] = []

        if not self.results_root.exists():
            return scenarios

        for scenario_dir in self.results_root.iterdir():
            if not scenario_dir.is_dir():
                continue

            error: str | None = None
            scenario_name: str = scenario_dir.name
            test_runs: list[TestRun] = []
            timestamp: datetime = datetime.now()

            # Format: {scenario_name}_{timestamp}
            dir_name = scenario_dir.name
            if "_" in dir_name:
                parts = dir_name.rsplit("_", 2)  # Split on last 2 underscores for date_time
                if len(parts) >= 3:
                    scenario_name = "_".join(parts[:-2])
                    date_part = parts[-2]
                    time_part = parts[-1]
                    try:
                        timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S")
                    except ValueError:
                        error = f"Not a scenario directory: {scenario_dir.absolute()}"
                else:
                    error = f"Not a scenario directory: {scenario_dir.absolute()}"
            else:
                error = f"Not a scenario directory: {scenario_dir.absolute()}"

            if not error:
                try:
                    test_runs = self._get_test_runs(scenario_dir)
                except Exception as e:
                    error = str(e)

            scenarios.append(
                TestScenarioInfo(
                    id=scenario_dir.name,
                    name=scenario_name,
                    timestamp=timestamp,
                    test_runs=test_runs,
                    error=error,
                )
            )

        # Sort by timestamp, newest first
        scenarios.sort(key=lambda x: x.timestamp, reverse=True)
        return scenarios

    def _get_test_runs(self, scenario_dir: Path) -> list[TestRun]:
        """Get test runs for a scenario."""
        test_runs = []

        system = SlurmSystem(
            name="slurm", install_path=Path("/"), output_path=Path("/"), default_partition="default", partitions=[]
        )

        for tr_dump in scenario_dir.rglob("test-run.toml"):
            trd = TestRunDetails.model_validate(toml.load(tr_dump))
            test_run = trd.to_test_run(system)
            test_run.output_path = tr_dump.parent
            test_runs.append(test_run)

        return test_runs
