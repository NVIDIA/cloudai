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

import os
from typing import Iterator
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cloudai import BaseJob, BaseJobWithOutput, FileOutputSystem, OutputType


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
def file_output_system():
    """Fixture to create a MockFileOutputSystem object for testing."""
    return MockFileOutputSystem("TestSystem", "scheduler", "/fake/system/output")


@pytest.mark.parametrize(
    "output_type, multiple_files, line_by_line",
    [
        (OutputType.STDOUT, False, False),
        (OutputType.STDERR, False, False),
        (OutputType.STDOUT, True, False),
        (OutputType.STDERR, True, False),
        (OutputType.STDOUT, False, True),
        (OutputType.STDERR, False, True),
        (OutputType.STDOUT, True, True),
        (OutputType.STDERR, True, True),
    ],
)
def test_retrieve_output_streams_single_file(file_output_system, mock_job, output_type, multiple_files, line_by_line):
    """Test retrieving output from a single file for both stdout and stderr."""
    mock_file_content = "This is a test output."
    file_name = "stdout.txt" if output_type == OutputType.STDOUT else "stderr.txt"
    file_path = os.path.join(mock_job.output_path, file_name)

    with (
        patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file,
        patch("os.path.isfile", return_value=True),
    ):
        result = file_output_system.retrieve_output_streams(
            mock_job, output_type, multiple_files=multiple_files, line_by_line=line_by_line
        )

    if not multiple_files:
        if line_by_line:
            assert isinstance(result, Iterator)
            assert list(result) == mock_file_content.splitlines()
        else:
            assert result == mock_file_content
        mock_file.assert_called_once_with(file_path, "r")
    else:
        assert result is None
        mock_file.assert_not_called()


@pytest.mark.parametrize(
    "output_type, multiple_files, line_by_line",
    [
        (OutputType.STDOUT, True, False),
        (OutputType.STDERR, True, False),
        (OutputType.STDOUT, True, True),
        (OutputType.STDERR, True, True),
    ],
)
def test_retrieve_output_streams_multiple_files(
    file_output_system, mock_job, output_type, multiple_files, line_by_line
):
    """Test retrieving output from multiple files for both stdout and stderr."""
    mock_file_content_1 = "This is the first part of the output."
    mock_file_content_2 = "This is the second part of the output."
    file_prefix = "stdout" if output_type == OutputType.STDOUT else "stderr"
    file_paths = [
        os.path.join(mock_job.output_path, f"{file_prefix}-0.txt"),
        os.path.join(mock_job.output_path, f"{file_prefix}-1.txt"),
    ]

    with patch("glob.glob", return_value=file_paths), patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = [
            mock_open(read_data=mock_file_content_1).return_value,
            mock_open(read_data=mock_file_content_2).return_value,
        ]

        result = file_output_system.retrieve_output_streams(
            mock_job, output_type, multiple_files=multiple_files, line_by_line=line_by_line
        )

    if multiple_files:
        assert isinstance(result, list)
        assert len(result) == 2
        if line_by_line:
            assert list(result[0]) == mock_file_content_1.splitlines()
            assert list(result[1]) == mock_file_content_2.splitlines()
        else:
            assert result[0] == mock_file_content_1
            assert result[1] == mock_file_content_2

    for path in file_paths:
        mock_file.assert_any_call(path, "r")


@pytest.mark.parametrize(
    "output_type, multiple_files",
    [
        (OutputType.STDOUT, False),
        (OutputType.STDERR, False),
        (OutputType.STDOUT, True),
        (OutputType.STDERR, True),
    ],
)
def test_retrieve_output_streams_no_files(file_output_system, mock_job, output_type, multiple_files):
    """Test retrieving output when no files are found for both stdout and stderr."""
    with patch("os.path.isfile", return_value=False):
        result = file_output_system.retrieve_output_streams(mock_job, output_type, multiple_files=multiple_files)
        assert result is None

    with patch("glob.glob", return_value=[]):
        result = file_output_system.retrieve_output_streams(mock_job, output_type, multiple_files=True)
        assert result is None


def test_retrieve_output_streams_mixed_files(file_output_system, mock_job):
    """Test retrieving a mix of single and multiple output files for both stdout and stderr."""
    mock_stdout_content = "This is stdout content."
    mock_stderr_content_1 = "This is the first stderr part."
    mock_stderr_content_2 = "This is the second stderr part."
    stdout_file = os.path.join(mock_job.output_path, "stdout.txt")
    stderr_files = [
        os.path.join(mock_job.output_path, "stderr-0.txt"),
        os.path.join(mock_job.output_path, "stderr-1.txt"),
    ]

    with (
        patch("builtins.open", mock_open(read_data=mock_stdout_content)) as mock_file,
        patch("os.path.isfile", return_value=True),
        patch("glob.glob", return_value=stderr_files),
    ):
        mock_file.side_effect = [
            mock_open(read_data=mock_stdout_content).return_value,
            mock_open(read_data=mock_stderr_content_1).return_value,
            mock_open(read_data=mock_stderr_content_2).return_value,
        ]

        stdout_result = file_output_system.retrieve_output_streams(
            mock_job, OutputType.STDOUT, multiple_files=False, line_by_line=False
        )
        stderr_result = file_output_system.retrieve_output_streams(
            mock_job, OutputType.STDERR, multiple_files=True, line_by_line=False
        )

    assert stdout_result == mock_stdout_content
    assert isinstance(stderr_result, list)
    assert stderr_result[0] == mock_stderr_content_1
    assert stderr_result[1] == mock_stderr_content_2

    mock_file.assert_any_call(stdout_file, "r")
    for path in stderr_files:
        mock_file.assert_any_call(path, "r")


@pytest.mark.parametrize(
    "output_type, mock_file_content",
    [
        (OutputType.STDOUT, "This is stdout content."),
        (OutputType.STDERR, "This is stderr content."),
    ],
)
def test_retrieve_output_streams_with_non_existent_file(file_output_system, mock_job, output_type, mock_file_content):
    """Test behavior when the expected output file does not exist."""

    with (
        patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file,
        patch("os.path.isfile", return_value=False),
    ):
        result = file_output_system.retrieve_output_streams(
            mock_job, output_type, multiple_files=False, line_by_line=False
        )

    assert result is None
    mock_file.assert_not_called()
