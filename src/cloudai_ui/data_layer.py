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

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel, ConfigDict

from cloudai.core import TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.reporter import SlurmReportItem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data
from cloudai.workloads.nixl_bench import NIXLBenchTestDefinition


class DSEDetails(BaseModel):
    """Details about a DSE run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    action: str
    reward: float
    observation: list[float]


@dataclass
class DataQuery:
    """Query parameters for loading dashboard data."""

    test_type: str | None  # None means all test types
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
    dse: DSEDetails | None = None

    @property
    def label(self) -> str:
        return f"{self.scenario_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def system_name(self) -> str:
        return self.test_run.test.test_template.system.name

    @property
    def dse_id(self) -> str:
        return f"{self.scenario_name} | {self.test_run.name} | {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def test_type(self) -> str:
        return self.test_run.test.test_definition.test_template_name.lower()


class DataProvider(ABC):
    """Abstract interface for dashboard data loading."""

    @abstractmethod
    def query_data(self, query: DataQuery) -> list[Record]:
        """Load data based on query parameters."""


class LocalFileDataProvider(DataProvider):
    """Local file-based data provider with lazy loading."""

    def __init__(self, results_root: Path):
        self.results_root = results_root
        self.issues: list[str] = []

    def query_data(self, query: DataQuery) -> list[Record]:
        """Load and filter data based on query parameters."""
        self.issues = []

        records: list[Record] = []
        for record in self.load_all_records():
            if query.scenario_names and (record.scenario_name not in query.scenario_names):
                continue

            if query.time_range_days and (record.timestamp < datetime.now() - timedelta(days=query.time_range_days)):
                continue

            if query.test_type is not None and not record.test_type.startswith(query.test_type.lower()):
                continue

            records.append(record)

        return records

    def load_all_records(self) -> list[Record]:
        records: list[Record] = []
        if not self.results_root.exists():
            self.issues.append(f"dir={self.results_root.absolute()}: does not exist")
            return records

        for scenario_dir in self.results_root.iterdir():
            if not scenario_dir.is_dir():
                self.issues.append(f"dir={scenario_dir.absolute()}: is not a directory")
                continue

            try:
                scenario_name, timestamp = self._parse_scenario_dir_name(scenario_dir.name)
            except ValueError as e:
                self.issues.append(f"dir={scenario_dir.absolute()}: {e}")
                continue

            records.extend(self._parse_one_scenario(scenario_dir, scenario_name, timestamp))

        return records

    def _parse_scenario_dir_name(self, dir_name: str) -> tuple[str, datetime]:
        # Format: {scenario_name}_{timestamp}
        if "_" not in dir_name:
            raise ValueError("no '_' in name")

        parts = dir_name.rsplit("_", 2)  # Split on last 2 underscores for date_time
        if len(parts) != 3:
            raise ValueError(f"expected 3 parts in name, got {len(parts)}")
        scenario_name = "_".join(parts[:-2])
        date_part = parts[-2]
        time_part = parts[-1]
        try:
            timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S")
        except ValueError as e:
            raise ValueError(f"expected timestamp in name, got {date_part}_{time_part} ({e})") from e

        return scenario_name, timestamp

    def _parse_one_scenario(self, scenario_dir: Path, scenario_name: str, timestamp: datetime) -> list[Record]:
        metadata_dirs = list(scenario_dir.rglob("metadata")) or [scenario_dir]
        metadata = SlurmReportItem.get_metadata(metadata_dirs[0].parent, self.results_root)
        if not metadata:
            self.issues.append(f"dir={scenario_dir.absolute()}: no metadata found")

        system = SlurmSystem(
            name=metadata.slurm.cluster_name if metadata else "unknown",
            install_path=Path("/"),
            output_path=Path("/"),
            default_partition="default",
            partitions=[],
        )

        records: list[Record] = []
        for test_dir in scenario_dir.iterdir():
            if not test_dir.is_dir():
                continue

            for test_iter_dir in test_dir.iterdir():
                try:
                    int(test_iter_dir.name)
                except ValueError:
                    continue

                if not test_iter_dir.is_dir():
                    continue

                records.extend(self._parse_single_test_iter(test_iter_dir, system, scenario_name, timestamp))

        return records

    def _parse_single_test_iter(
        self, test_iter_dir: Path, system: SlurmSystem, scenario_name: str, timestamp: datetime
    ) -> list[Record]:
        records: list[Record] = []

        trajectory_file, trajectory_data = test_iter_dir / "trajectory.csv", pd.DataFrame()
        if trajectory_file.exists():
            trajectory_data = pd.read_csv(trajectory_file)

        for tr_dump in test_iter_dir.rglob("test-run.toml"):
            try:
                trd = TestRunDetails.model_validate(toml.load(tr_dump))
            except Exception as e:
                self.issues.append(f"dir={test_iter_dir.absolute()}: {e}")
                continue
            test_run = trd.to_test_run(system)
            test_run.output_path = tr_dump.parent

            dse_details = None
            if not trajectory_data.empty:
                try:
                    step_data = trajectory_data[trajectory_data["step"] == test_run.step].iloc[0]
                    dse_details = DSEDetails(
                        step=test_run.step,
                        action=step_data["action"],
                        reward=step_data["reward"],
                        observation=json.loads(step_data["observation"]),
                    )
                except Exception as e:
                    self.issues.append(f"dir={test_iter_dir.absolute()}: step={test_run.step} {e}")
                    continue

            df = pd.DataFrame()
            if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                df = extract_nccl_data_as_df(test_run)
            elif isinstance(test_run.test.test_definition, NIXLBenchTestDefinition):
                df = extract_nixl_data_as_df(test_run)

            records.append(
                Record(test_run=test_run, df=df, scenario_name=scenario_name, timestamp=timestamp, dse=dse_details)
            )

        return records


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
