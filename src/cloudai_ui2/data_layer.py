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

from cloudai.core import TestRun
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data
from cloudai_ui.data_layer import LocalFileDataProvider as _LocalFileDataProvider
from cloudai_ui.data_layer import TestScenarioInfo


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
        nccl_data: list[Record] = []
        for scenario in self.filtered_scenarios(query):
            for test_run in scenario.test_runs:
                if not isinstance(test_run.test.test_definition, NCCLTestDefinition):
                    continue

                df = extract_nccl_data_as_df(test_run)
                if not df.empty:
                    nccl_data.append(
                        Record(test_run=test_run, df=df, scenario_name=scenario.name, timestamp=scenario.timestamp)
                    )

        return nccl_data

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
