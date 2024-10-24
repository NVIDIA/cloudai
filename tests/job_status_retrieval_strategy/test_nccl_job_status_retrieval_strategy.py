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

from cloudai.schema.test_template.nccl_test.job_status_retrieval_strategy import NcclTestJobStatusRetrievalStrategy


class TestNcclTestJobStatusRetrievalStrategy:
    def setup_method(self) -> None:
        self.js = NcclTestJobStatusRetrievalStrategy()

    def test_no_stdout_file(self, tmp_path: Path) -> None:
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"stdout.txt file not found in the specified output directory {tmp_path}. "
            "This file is expected to be created as a result of the NCCL test run. "
            "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
            "You can run the generated NCCL test command manually and verify the creation of "
            f"{tmp_path / 'stdout.txt'}. If the issue persists, contact the system administrator."
        )

    def test_successful_job(self, tmp_path: Path) -> None:
        stdout_file = tmp_path / "stdout.txt"
        stdout_content = """
        # Some initialization output
        # More output
        # Out of bounds values : 0 OK
        # Avg bus bandwidth    : 100.00
        # Some final output
        """
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(tmp_path)
        assert result.is_successful
        assert result.error_message == ""

    def test_failed_job(self, tmp_path: Path) -> None:
        stdout_file = tmp_path / "stdout.txt"
        stdout_content = """
        # Some initialization output
        # More output
        # Some final output without success indicators
        """
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"Missing success indicators in {stdout_file}: '# Out of bounds values', '# Avg bus bandwidth'. "
            "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
            f"Please review the NCCL test output and errors in the file. "
            "Ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
            f"and check if {stdout_file} is created and contains the expected keywords. "
            "If the issue persists, contact the system administrator."
        )

    def test_nccl_failure_job(self, tmp_path: Path) -> None:
        stdout_file = tmp_path / "stdout.txt"
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
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"NCCL test failure detected in {stdout_file}. "
            "Possible reasons include network errors or remote process exits. "
            "Please review the NCCL test output and errors in the file first. "
            "If the issue persists, contact the system administrator."
        )

    def test_generic_test_failure_job(self, tmp_path: Path) -> None:
        stdout_file = tmp_path / "stdout.txt"
        stdout_content = """
        # Some initialization output
        .. node pid: Test failure common.cu:401
        """
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"Test failure detected in {stdout_file}. "
            "Please review the specific test failure messages in the file. "
            "Ensure that the NCCL test environment is correctly set up and configured. "
            "If the issue persists, contact the system administrator."
        )
