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

"""Data layer for CloudAI UI v2 with lazy loading support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import toml

from cloudai.core import TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.reporter import SlurmReportItem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data
from cloudai.workloads.nixl_bench import NIXLBenchTestDefinition


@dataclass
class TestScenarioInfo:
    """Information about a test scenario run."""

    id: str
    name: str
    timestamp: datetime
    test_runs: list[TestRun]
    error: str | None = None


@dataclass
class DataQuery:
    """Query parameters for loading dashboard data."""

    test_type: str
    time_range_days: int = 7
    scenario_names: list[str] | None = None


@dataclass(frozen=True)
class Record:
    """
    Immutable DB-like test run data.

    Includes the test run, the extracted results as a DataFrame, the scenario name, and the timestamp. This is what is
    stored in a database for each run. Data providers return a list of such records.
    """

    test_run: TestRun
    df: pd.DataFrame
    scenario_name: str
    timestamp: datetime

    @property
    def label(self) -> str:
        return f"{self.scenario_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def system_name(self) -> str:
        return self.test_run.test.test_template.system.name


class _BaseFileDataProvider(ABC):
    """Abstract base class for file-based data providers."""

    @abstractmethod
    def get_scenarios(self) -> list[TestScenarioInfo]:
        """Get list of all test scenarios."""
        pass


class _LocalFileDataProvider(_BaseFileDataProvider):
    """Internal data provider that reads from local filesystem."""

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
                    except ValueError as e:
                        error = f"Not a scenario directory 3: {scenario_dir.absolute()}: {e} {parts}"
                else:
                    error = f"Not a scenario directory 2: {scenario_dir.absolute()}"
            else:
                error = f"Not a scenario directory 1: {scenario_dir.absolute()}"

            metadata = SlurmReportItem.get_metadata(scenario_dir, self.results_root)
            system = SlurmSystem(
                name=metadata.slurm.cluster_name if metadata else "unknown",
                install_path=Path("/"),
                output_path=Path("/"),
                default_partition="default",
                partitions=[],
            )
            if not error:
                try:
                    test_runs = self._get_test_runs(scenario_dir, system)
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

    def _get_test_runs(self, scenario_dir: Path, system: SlurmSystem) -> list[TestRun]:
        """Get test runs for a scenario."""
        test_runs = []

        for tr_dump in scenario_dir.rglob("test-run.toml"):
            trd = TestRunDetails.model_validate(toml.load(tr_dump))
            test_run = trd.to_test_run(system)
            test_run.output_path = tr_dump.parent
            test_runs.append(test_run)

        return test_runs


class DataProvider(ABC):
    """Abstract interface for dashboard data loading."""

    @abstractmethod
    def query_data(self, query: DataQuery) -> list[Record]:
        """Load data based on query parameters."""


class LocalFileDataProvider(DataProvider):
    """Local file-based data provider with lazy loading."""

    def __init__(self, results_root: Path):
        self.results_root = results_root
        self._file_provider = _LocalFileDataProvider(results_root)

    def query_data(self, query: DataQuery) -> list[Record]:
        """Load and filter data based on query (ignores time range for local files)."""
        records: list[Record] = []
        for scenario in self.filtered_scenarios(query):
            for test_run in scenario.test_runs:
                df = pd.DataFrame()

                if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                    df = extract_nccl_data_as_df(test_run)
                elif isinstance(test_run.test.test_definition, NIXLBenchTestDefinition):
                    df = extract_nixl_data_as_df(test_run)

                if not df.empty:
                    records.append(
                        Record(test_run=test_run, df=df, scenario_name=scenario.name, timestamp=scenario.timestamp)
                    )

        return records

    def filtered_scenarios(self, query: DataQuery) -> list[TestScenarioInfo]:
        filtered_scenarios: list[TestScenarioInfo] = []
        for scenario in self._file_provider.get_scenarios():
            # Filter by scenario name if specified
            if query.scenario_names and scenario.name not in query.scenario_names:
                continue
            if query.time_range_days and scenario.timestamp < datetime.now() - timedelta(days=query.time_range_days):
                continue

            # Filter test runs by test type
            matching_runs = [
                tr
                for tr in scenario.test_runs
                if tr.test.test_definition.test_template_name.lower().startswith(query.test_type.lower())
            ]

            if matching_runs:
                # Create filtered scenario with only matching test runs
                filtered_scenario = TestScenarioInfo(
                    id=scenario.id,
                    name=scenario.name,
                    timestamp=scenario.timestamp,
                    test_runs=matching_runs,
                    error=scenario.error,
                )
                filtered_scenarios.append(filtered_scenario)

        return filtered_scenarios


def extract_nixl_data_as_df(tr: TestRun) -> pd.DataFrame:
    if (tr.output_path / "nixlbench.csv").exists():
        return pd.read_csv(tr.output_path / "nixlbench.csv")
    return pd.DataFrame(
        {
            "block_size": pd.Series([], dtype=int),
            "batch_size": pd.Series([], dtype=int),
            "avg_lat": pd.Series([], dtype=float),
            "bw_gb_sec": pd.Series([], dtype=float),
        }
    )


def extract_nccl_data_as_df(test_run: TestRun) -> pd.DataFrame:
    stdout_path = test_run.output_path / "stdout.txt"

    if not stdout_path.exists():
        return pd.DataFrame()

    parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = extract_nccl_data(stdout_path)
    if not parsed_data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        parsed_data_rows,
        columns=[
            "Size (B)",
            "Count",
            "Type",
            "Redop",
            "Root",
            "Time (us) Out-of-place",
            "Algbw (GB/s) Out-of-place",
            "Busbw (GB/s) Out-of-place",
            "#Wrong Out-of-place",
            "Time (us) In-place",
            "Algbw (GB/s) In-place",
            "Busbw (GB/s) In-place",
            "#Wrong In-place",
        ],
    )

    df["GPU Type"] = gpu_type
    df["Devices per Node"] = num_devices_per_node
    df["Ranks"] = num_ranks

    df["Size (B)"] = df["Size (B)"].astype(int)
    df["Time (us) Out-of-place"] = df["Time (us) Out-of-place"].astype(float).round(2)
    df["Time (us) In-place"] = df["Time (us) In-place"].astype(float).round(2)
    df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
    df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
    df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
    df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)

    return df
