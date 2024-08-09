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


from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cloudai.runner.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone_system import StandaloneSystem


@pytest.fixture
def standalone_system():
    """
    Fixture to create a StandaloneSystem instance for testing.

    Returns:
        StandaloneSystem: A new instance of StandaloneSystem for testing.
    """
    return StandaloneSystem("StandaloneTestSystem", Path("/fake/output/path"))


@pytest.fixture
def mock_test():
    """
    Fixture to create a mock Test instance for testing.

    Returns:
        MagicMock: A mocked Test instance.
    """
    return MagicMock(name="MockTest")


@pytest.fixture
def standalone_job(standalone_system, mock_test):
    """
    Fixture to create a StandaloneJob instance for testing.

    Args:
        standalone_system (StandaloneSystem): The system where the job will be executed.
        mock_test (Test): The mock test instance associated with the job.

    Returns:
        StandaloneJob: A new instance of StandaloneJob for testing.
    """
    return StandaloneJob("run", standalone_system, mock_test, 12345, Path("/fake/output/path"))


@pytest.mark.parametrize(
    "ps_output, expected_result",
    [
        ("12345\n", True),  # Job is running, PID is in ps output
        ("", False),  # Job is not running, ps output is empty
    ],
)
@patch("cloudai.util.CommandShell.execute")
def test_is_job_running(mock_execute, standalone_system, standalone_job, ps_output, expected_result):
    """
    Test if a job is running using a mocked CommandShell.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        standalone_job (StandaloneJob): Job instance to check.
        ps_output (str): Mocked output of the ps command.
        expected_result (bool): Expected result for the job running status.
    """
    mock_process = MagicMock()
    mock_process.communicate.return_value = (ps_output, "")
    mock_execute.return_value = mock_process

    assert standalone_system.is_job_running(standalone_job) == expected_result


@patch("cloudai.util.CommandShell.execute")
def test_kill_job(mock_execute, standalone_system, standalone_job):
    """
    Test if a job can be killed using a mocked CommandShell.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        standalone_job (StandaloneJob): Job instance to kill.
    """
    mock_process = MagicMock()
    mock_execute.return_value = mock_process

    standalone_system.kill(standalone_job)
    kill_command = f"kill -9 {standalone_job.get_id()}"

    mock_execute.assert_called_once_with(kill_command)


def test_repr(standalone_system):
    """
    Test the string representation of StandaloneSystem.

    Args:
        standalone_system (StandaloneSystem): Instance of the system under test.
    """
    expected_repr = f"StandaloneSystem(name={standalone_system.name}, scheduler=standalone)"
    assert repr(standalone_system) == expected_repr
