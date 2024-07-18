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
import re
from pathlib import Path

from cloudai import JobStatusResult, JobStatusRetrievalStrategy


class JaxToolboxJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for JaxToolbox by checking the contents of output_path."""

    def get_job_status(self, output_path: str) -> JobStatusResult:
        """
        Determine the job status by examining 'profile_stderr.txt' and 'error-*.txt' in the output directory.

        Args:
            output_path (str): Path to the directory containing 'profile_stderr.txt' and 'error-*.txt'.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        profile_stderr_path = os.path.join(output_path, "profile_stderr.txt")

        result = self.check_profile_stderr(profile_stderr_path, output_path)
        if not result.is_successful:
            return result

        error_files = list(Path(output_path).glob("error-*.txt"))
        if not error_files:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"No 'error-*.txt' files found in the output directory, {output_path}. There are two stages in the "
                    "Grok run. The profiling stage passed successfully, but something went wrong in the actual run "
                    "stage. Please ensure the actual run stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        return self.check_error_files(error_files, output_path)

    def check_profile_stderr(self, profile_stderr_path: str, output_path: str) -> JobStatusResult:
        """
        Check the profile_stderr.txt file for known error messages.

        Args:
            profile_stderr_path (str): Path to the 'profile_stderr.txt' file.
            output_path (str): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        if not os.path.isfile(profile_stderr_path):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"profile_stderr.txt file not found in the specified output directory, {output_path}. "
                    "This file is expected to be created during the profiling stage. "
                    "Please ensure the profiling stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        with open(profile_stderr_path, "r") as file:
            content = file.read()

            if "[PAX STATUS]: E2E time: Elapsed time for " not in content:
                return JobStatusResult(
                    is_successful=False,
                    error_message=(
                        "The profiling stage completed but did not generate the expected '[PAX STATUS]: E2E time: "
                        "Elapsed time for ' keyword. There are two stages in the Grok run, and an error occurred in "
                        "the profiling stage. While profile_stderr.txt was created, the expected keyword is missing. "
                        "You need to run the sbatch script manually to see what happens."
                    ),
                )

            result = self.check_common_errors(content, profile_stderr_path, output_path)
            if not result.is_successful:
                return result

        return JobStatusResult(is_successful=True)

    def check_common_errors(self, content: str, file_path: str, output_path: str) -> JobStatusResult:
        """
        Check for common errors in the file content.

        Args:
            content (str): The content of the file to check.
            file_path (str): The path of the file being checked.
            output_path (str): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        if "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
                    "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
                    "First, check if GPUs are available on the server. "
                    "Second, if running the job with Slurm, ensure proper resource-related options are set, "
                    "including GPU resource requirements. Lastly, check environment variables. "
                    "If the problem persists, verify commands and environment variables by running a simple GPU-only "
                    "example command."
                ),
            )
        if "Terminating process because the coordinator detected missing heartbeats" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "Terminating process because the coordinator detected missing heartbeats. This most likely "
                    f"indicates that another task died. Please review the file at {file_path} and any relevant logs in"
                    f" {output_path}. Ensure the servers allocated for this task can reach each other with their "
                    "hostnames, and they can open any ports and reach others' ports."
                ),
            )
        if "NCCL operation ncclGroupEnd() failed" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
                    "passes. Run with NCCL_DEBUG=INFO for more details."
                ),
            )
        if re.search(r"pyxis:\s+mktemp: failed to create directory via template", content):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "pyxis: mktemp: failed to create directory via template. This is due to insufficient disk cache "
                    "capacity. This is not a CloudAI issue. When you run JaxToolbox, CloudAI executes srun, which "
                    "includes the container image option. When the container image argument is a remote URL, "
                    "Slurm downloads and caches the Docker image locally. It fails with this error when the system "
                    "does not have enough disk capacity to cache the Docker image."
                ),
            )

        return JobStatusResult(is_successful=True)

    def check_error_files(self, error_files: list, output_path: str) -> JobStatusResult:
        """
        Check the error-*.txt files for known error messages.

        Args:
            error_files (list): List of paths to error files.
            output_path (str): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        for error_file in error_files:
            with open(error_file, "r") as file:
                content = file.read()
                result = self.check_common_errors(content, error_file, output_path)
                if not result.is_successful:
                    return result
                if "E2E time: Elapsed time for" not in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"The file {error_file} does not contain the expected 'E2E time: Elapsed time for' "
                            "keyword at the end. This indicates the actual run did not complete successfully. "
                            "Please debug this manually to ensure the actual run stage completes as expected."
                        ),
                    )

        return JobStatusResult(is_successful=True)
