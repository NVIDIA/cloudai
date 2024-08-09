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


from cloudai import BaseJob, JobStatusResult, JobStatusRetrievalStrategy, OutputType, System


class NcclTestJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for NCCL tests by checking the contents of stdout."""

    def get_job_status(self, system: System, job: BaseJob) -> JobStatusResult:
        """
        Determine the job status by examining the stdout retrieved from the system.

        Args:
            system (System): The system object used to retrieve stdout.
            job (BaseJob): The job object associated with the test.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        # Retrieve stdout content through the system's API
        stdout_content = system.retrieve_output_streams(
            job, OutputType.STDOUT, multiple_files=False, line_by_line=False
        )

        if stdout_content is None:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "stdout not found. This output is expected as a result of the NCCL test run. "
                    "Please ensure the NCCL test was executed properly and that stdout was generated. "
                    "If the issue persists, contact the system administrator."
                ),
            )

        # Check for specific error patterns
        if "Test NCCL failure" in stdout_content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "NCCL test failure detected in stdout. Possible reasons include network errors or "
                    "remote process exits. Please review the NCCL test output and errors first. "
                    "If the issue persists, contact the system administrator."
                ),
            )
        if "Test failure" in stdout_content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "Test failure detected in stdout. Please review the specific test failure messages. "
                    "Ensure that the NCCL test environment is correctly set up and configured. "
                    "If the issue persists, contact the system administrator."
                ),
            )

        # Check for success indicators
        if "# Out of bounds values" in stdout_content and "# Avg bus bandwidth" in stdout_content:
            return JobStatusResult(is_successful=True)

        # Identify missing success indicators
        missing_indicators = []
        if "# Out of bounds values" not in stdout_content:
            missing_indicators.append("'# Out of bounds values'")
        if "# Avg bus bandwidth" not in stdout_content:
            missing_indicators.append("'# Avg bus bandwidth'")

        error_message = (
            f"Missing success indicators in stdout: {', '.join(missing_indicators)}. "
            "These keywords are expected to be present in the output, usually towards the end. "
            "Please review the NCCL test output. Ensure the NCCL test ran to completion. "
            "If the issue persists, contact the system administrator."
        )
        return JobStatusResult(is_successful=False, error_message=error_message)
