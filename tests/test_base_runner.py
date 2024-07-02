#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from datetime import datetime
from unittest.mock import patch

import pytest
from cloudai import BaseRunner


@pytest.fixture
def mock_datetime_now():
    with patch("cloudai._core.base_runner.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.strftime = datetime.strftime
        yield mock_datetime


def test_setup_output_directory(mock_datetime_now, tmp_path):  # noqa
    scenario_name = "test_scenario"
    base_output_path = tmp_path / "base_output_path"
    expected_time_str = "2024-01-01_12-00-00"
    expected_path = base_output_path / f"{scenario_name}_{expected_time_str}"

    output_path = BaseRunner.setup_output_directory(scenario_name, str(base_output_path))

    print(f"Expected path: {expected_path}, Output path: {output_path}")

    assert os.path.exists(base_output_path)
    assert os.path.exists(expected_path)
    assert output_path == str(expected_path)


def test_setup_output_directory_existing_base_path(mock_datetime_now, tmp_path):  # noqa
    scenario_name = "test_scenario"
    base_output_path = tmp_path / "base_output_path"
    expected_time_str = "2024-01-01_12-00-00"
    expected_path = base_output_path / f"{scenario_name}_{expected_time_str}"

    base_output_path.mkdir()
    output_path = BaseRunner.setup_output_directory(scenario_name, str(base_output_path))

    print(f"Expected path: {expected_path}, Output path: {output_path}")

    assert os.path.exists(expected_path)
    assert output_path == str(expected_path)


def test_setup_output_directory_handles_oserror(mock_datetime_now, tmp_path):  # noqa
    scenario_name = "test_scenario"
    base_output_path = tmp_path / "base_output_path"

    # Simulate a permission error by setting the directory to read-only
    base_output_path.mkdir()
    os.chmod(base_output_path, 0o400)

    with pytest.raises(PermissionError):
        BaseRunner.setup_output_directory(scenario_name, str(base_output_path))

    # Reset permissions so that pytest can clean up the directory
    os.chmod(base_output_path, 0o700)
