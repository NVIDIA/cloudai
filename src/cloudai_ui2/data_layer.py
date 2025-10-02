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
from typing import List

from cloudai_ui.data_layer import LocalFileDataProvider as _LocalFileDataProvider
from cloudai_ui.data_layer import TestScenarioInfo


@dataclass
class DataQuery:
    """Query parameters for loading dashboard data."""

    test_type: str
    time_range_days: int = 7
    scenario_names: list[str] | None = None


class DataProvider(ABC):
    """Abstract interface for dashboard data loading."""

    @abstractmethod
    def query_data(self, query: DataQuery) -> List[TestScenarioInfo]:
        """Load data based on query parameters."""


class LocalFileDataProvider(DataProvider):
    """Local file-based data provider with lazy loading."""

    def __init__(self, results_root: Path):
        self.results_root = results_root
        self._file_provider = _LocalFileDataProvider(results_root)

    def query_data(self, query: DataQuery) -> List[TestScenarioInfo]:
        """Load and filter data based on query (ignores time range for local files)."""
        all_scenarios = self._file_provider.get_scenarios()

        filtered_scenarios = []
        for scenario in all_scenarios:
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
