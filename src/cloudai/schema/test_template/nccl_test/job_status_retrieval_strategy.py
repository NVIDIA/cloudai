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

from cloudai import JobStatusResult, JobStatusRetrievalStrategy


class NcclTestJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for NCCL tests by checking the contents of 'stdout.txt'."""

    def get_job_status(self, output_path: str) -> JobStatusResult:
        """
        Determine the job status by examining 'stdout.txt' in the output directory.

        Args:
            output_path (str): Path to the directory containing 'stdout.txt'.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        stdout_path = os.path.join(output_path, "stdout.txt")
        if os.path.isfile(stdout_path):
            with open(stdout_path, "r") as file:
                content = file.read()

                # Check for specific error patterns
                if "Test NCCL failure" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"NCCL test failure detected in {stdout_path}. "
                            "Possible reasons include network errors or remote process exits. "
                            "Please review the NCCL test output and errors in the file first. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )
                if "Test failure" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"Test failure detected in {stdout_path}. "
                            "Please review the specific test failure messages in the file. "
                            "Ensure that the NCCL test environment is correctly set up and configured. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )

                # Check for success indicators
                if "# Out of bounds values" in content and "# Avg bus bandwidth" in content:
                    return JobStatusResult(is_successful=True)

                # Identify missing success indicators
                missing_indicators = []
                if "# Out of bounds values" not in content:
                    missing_indicators.append("'# Out of bounds values'")
                if "# Avg bus bandwidth" not in content:
                    missing_indicators.append("'# Avg bus bandwidth'")

                error_message = (
                    f"Missing success indicators in {stdout_path}: {', '.join(missing_indicators)}. "
                    "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
                    "Please review the NCCL test output and errors in the file. "
                    "Ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
                    f"and check if {stdout_path} is created and contains the expected keywords. "
                    "If the issue persists, contact the system administrator."
                )
                return JobStatusResult(is_successful=False, error_message=error_message)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stdout.txt file not found in the specified output directory {output_path}. "
                "This file is expected to be created as a result of the NCCL test run. "
                "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
                f"You can run the generated NCCL test command manually and verify the creation of {stdout_path}. "
                "If the issue persists, contact the system administrator."
            ),
        )
