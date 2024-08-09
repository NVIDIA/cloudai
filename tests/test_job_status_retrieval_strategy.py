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

from unittest.mock import MagicMock, patch

import pytest
from cloudai import BaseJob, BaseJobWithOutput, FileOutputSystem
from cloudai.schema.test_template.nccl_test.job_status_retrieval_strategy import NcclTestJobStatusRetrievalStrategy


class MockFileOutputSystem(FileOutputSystem):
    """A mock subclass to instantiate FileOutputSystem for testing."""

    def update(self) -> None:
        """Mock implementation of the abstract update method."""
        pass

    def is_job_running(self, job: "BaseJob") -> bool:
        """Mock implementation of the abstract is_job_running method."""
        return True

    def is_job_completed(self, job: "BaseJob") -> bool:
        """Mock implementation of the abstract is_job_completed method."""
        return True

    def kill(self, job: "BaseJob") -> None:
        """Mock implementation of the abstract kill method."""
        pass


@pytest.fixture
def mock_job():
    """Fixture to create a mock BaseJobWithOutput object."""
    job = MagicMock(spec=BaseJobWithOutput)
    job.output_path = "/fake/output/path"
    return job


@pytest.fixture
def mock_system():
    """Fixture to create a MockFileOutputSystem object."""
    system = MockFileOutputSystem(name="MockSystem", scheduler="mock_scheduler", output_path="/fake/system/output")
    return system


class TestNcclTestJobStatusRetrievalStrategy:
    """Tests for the NcclTestJobStatusRetrievalStrategy class."""

    def setup_method(self) -> None:
        """Setup method for initializing NcclTestJobStatusRetrievalStrategy."""
        self.js = NcclTestJobStatusRetrievalStrategy()

    def test_no_stdout_file(self, mock_system: FileOutputSystem, mock_job: BaseJobWithOutput) -> None:
        """Test that job status is False when no stdout is retrieved."""
        with patch.object(mock_system, "retrieve_output_streams", return_value=None):
            result = self.js.get_job_status(mock_system, mock_job)
            assert not result.is_successful
            assert result.error_message == (
                "stdout not found. This output is expected as a result of the NCCL test run. "
                "Please ensure the NCCL test was executed properly and that stdout was generated. "
                "If the issue persists, contact the system administrator."
            )

    def test_successful_job(self, mock_system: FileOutputSystem, mock_job: BaseJobWithOutput) -> None:
        """Test that job status is True when stdout contains success indicators."""
        stdout_content = """
        # Some initialization output
        # More output
        # Out of bounds values : 0 OK
        # Avg bus bandwidth    : 100.00
        # Some final output
        """
        with patch.object(mock_system, "retrieve_output_streams", return_value=stdout_content):
            result = self.js.get_job_status(mock_system, mock_job)
            assert result.is_successful
            assert result.error_message == ""

    def test_failed_job(self, mock_system: FileOutputSystem, mock_job: BaseJobWithOutput) -> None:
        """Test that job status is False when stdout does not contain success indicators."""
        stdout_content = """
        # Some initialization output
        # More output
        # Some final output without success indicators
        """
        with patch.object(mock_system, "retrieve_output_streams", return_value=stdout_content):
            result = self.js.get_job_status(mock_system, mock_job)
            assert not result.is_successful
            assert result.error_message == (
                "Missing success indicators in stdout: '# Out of bounds values', '# Avg bus bandwidth'. "
                "These keywords are expected to be present in the output, usually towards the end. "
                "Please review the NCCL test output. Ensure the NCCL test ran to completion. "
                "If the issue persists, contact the system administrator."
            )

    def test_nccl_failure_job(self, mock_system: FileOutputSystem, mock_job: BaseJobWithOutput) -> None:
        """Test that job status is False when stdout contains NCCL failure indicators."""
        stdout_content = """
        # Some initialization output
        node: Test NCCL failure common.cu:303 'remote process exited or there was a network error / '
        .. node pid: Test failure common.cu:401
        .. node pid: Test failure common.cu:588
        .. node pid: Test failure alltoall.cu:97
        .. node pid: Test failure common.cu:615
        .. node pid: Test failure common.cu:1019
        .. node pid: Test failure common.cu:844
        """
        with patch.object(mock_system, "retrieve_output_streams", return_value=stdout_content):
            result = self.js.get_job_status(mock_system, mock_job)
            assert not result.is_successful
            assert result.error_message == (
                "NCCL test failure detected in stdout. "
                "Possible reasons include network errors or remote process exits. "
                "Please review the NCCL test output and errors first. "
                "If the issue persists, contact the system administrator."
            )

    def test_generic_test_failure_job(self, mock_system: FileOutputSystem, mock_job: BaseJobWithOutput) -> None:
        """Test that job status is False when stdout contains generic test failure indicators."""
        stdout_content = """
        # Some initialization output
        .. node pid: Test failure common.cu:401
        """
        with patch.object(mock_system, "retrieve_output_streams", return_value=stdout_content):
            result = self.js.get_job_status(mock_system, mock_job)
            assert not result.is_successful
            assert result.error_message == (
                "Test failure detected in stdout. "
                "Please review the specific test failure messages. "
                "Ensure that the NCCL test environment is correctly set up and configured. "
                "If the issue persists, contact the system administrator."
            )
