# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.systems.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone.standalone_system import StandaloneSystem


@pytest.fixture
def standalone_system():
    """
    Fixture to create a StandaloneSystem instance for testing.

    Returns:
        StandaloneSystem: A new instance of StandaloneSystem for testing.
    """
    return StandaloneSystem(
        name="StandaloneTestSystem",
        install_path=Path("/fake/install/path"),
        output_path=Path("/fake/output/path"),
    )


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
    return StandaloneJob(mock_test, id=12345)


@pytest.mark.parametrize("returncode, expected_result", [(None, True), (0, False), (1, False)])
def test_is_job_running_polls_the_process_handle(standalone_system, mock_test, returncode, expected_result):
    """Completion is read from the handle we own, not discovered by running `ps`.

    Replaces a test that mocked ``CommandShell.execute`` and asserted the pid appeared in
    mocked ``ps`` stdout. That assertion described the old implementation -- it could only
    pass while completion was detected by shelling out -- so it could not survive this
    change. ``Popen.poll()`` returns ``None`` while the child runs and its exit status once
    it has finished.
    """
    process = MagicMock()
    process.poll.return_value = returncode
    job = StandaloneJob(mock_test, id=12345, process=process)

    assert standalone_system.is_job_running(job) is expected_result
    process.poll.assert_called()


def test_is_job_running_spawns_no_subprocess(standalone_system, mock_test):
    """The point of the change: a completion check must not cost a process.

    The old path spawned a shell and a ``ps`` on every monitor tick. With the poll interval
    driven down for a fast backend, that cost more than the job being waited for.
    """
    process = MagicMock()
    process.poll.return_value = None
    job = StandaloneJob(mock_test, id=12345, process=process)

    with patch("cloudai.util.CommandShell.execute") as mock_execute:
        assert standalone_system.is_job_running(job) is True

    mock_execute.assert_not_called()


@pytest.mark.parametrize("alive, expected_result", [(True, True), (False, False)])
def test_is_job_running_falls_back_to_a_signal_probe(standalone_system, mock_test, alive, expected_result):
    """A job with no handle was not launched by this process; probe instead of giving up."""
    job = StandaloneJob(mock_test, id=12345, process=None)

    def fake_kill(pid: int, sig: int) -> None:
        assert sig == 0, "probe must not deliver a signal"
        if not alive:
            raise ProcessLookupError

    with patch("os.kill", side_effect=fake_kill):
        assert standalone_system.is_job_running(job) is expected_result


def test_is_job_running_handles_a_non_numeric_id(standalone_system, mock_test):
    """Dry-run and reconstructed jobs can carry an id that is not a pid."""
    job = StandaloneJob(mock_test, id="not-a-pid", process=None)

    assert standalone_system.is_job_running(job) is False


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
    kill_command = f"kill -9 {standalone_job.id}"

    mock_execute.assert_called_once_with(kill_command)


@pytest.mark.parametrize("job_id", [0, "0", "00", "+0", "-0", -1, "-1", "not-a-pid"])
@patch("cloudai.util.CommandShell.execute")
def test_kill_job_skips_invalid_pid(mock_execute, standalone_system, mock_test, job_id):
    """
    Test that standalone dry-run sentinel IDs and invalid PIDs are not killed.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        mock_test (Test): The mock test instance associated with the job.
        job_id (int | str): Job ID that must not be killed.
    """
    job = StandaloneJob(mock_test, id=job_id)

    standalone_system.kill(job)

    mock_execute.assert_not_called()


@patch("cloudai.util.CommandShell.execute")
def test_kill_job_normalizes_numeric_pid(mock_execute, standalone_system, mock_test):
    """
    Test that standalone termination uses a validated numeric process ID.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        mock_test (Test): The mock test instance associated with the job.
    """
    job = StandaloneJob(mock_test, id="0012345")

    standalone_system.kill(job)

    mock_execute.assert_called_once_with("kill -9 12345")
