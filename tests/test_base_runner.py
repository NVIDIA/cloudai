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

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from cloudai import BaseJob, BaseRunner, System, TestScenario


class MockRunner(BaseRunner):
    def _submit_test(self, test):
        job_id = 1
        output_path = self.get_job_output_path(test)
        return BaseJob(job_id, test, output_path)

    def is_job_running(self, job):
        return False

    def is_job_completed(self, job):
        return True

    def kill_job(self, job):
        pass


@pytest.fixture
def mock_datetime_now():
    with patch("cloudai._core.base_runner.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.strftime = datetime.strftime
        yield mock_datetime


def test_setup_output_directory(mock_datetime_now, tmp_path):
    scenario_name = "test_scenario"
    base_output_path = tmp_path / "base_output_path"
    expected_time_str = "2024-01-01_12-00-00"
    expected_path = base_output_path / f"{scenario_name}_{expected_time_str}"

    # Mock TestScenario and System
    mock_test_scenario = MagicMock(spec=TestScenario)
    mock_test_scenario.name = scenario_name
    mock_system = MagicMock(spec=System)
    mock_system.output_path = str(base_output_path)
    mock_system.monitor_interval = 5

    runner = MockRunner("run", mock_system, mock_test_scenario)

    assert base_output_path.exists()
    assert expected_path.exists()
    assert runner.output_path == str(expected_path)


def test_setup_output_directory_existing_base_path(mock_datetime_now, tmp_path):
    scenario_name = "test_scenario"
    base_output_path = tmp_path / "base_output_path"
    expected_time_str = "2024-01-01_12-00-00"
    expected_path = base_output_path / f"{scenario_name}_{expected_time_str}"

    base_output_path.mkdir()

    # Mock TestScenario and System
    mock_test_scenario = MagicMock(spec=TestScenario)
    mock_test_scenario.name = scenario_name
    mock_system = MagicMock(spec=System)
    mock_system.output_path = str(base_output_path)
    mock_system.monitor_interval = 5

    runner = MockRunner("run", mock_system, mock_test_scenario)

    assert expected_path.exists()
    assert runner.output_path == str(expected_path)
