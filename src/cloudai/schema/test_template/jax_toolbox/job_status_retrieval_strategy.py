# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

from cloudai._core.job_status_result import JobStatusResult
from cloudai._core.job_status_retrieval_strategy import JobStatusRetrievalStrategy


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
        if not os.path.isfile(profile_stderr_path):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "profile_stderr.txt file not found in the specified output directory. "
                    "This file is expected to be created during the profiling stage of the Grok run. "
                    "Please ensure the profiling stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        with open(profile_stderr_path, "r") as file:
            content = file.read()
            if "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected" in content:
                return JobStatusResult(
                    is_successful=False,
                    error_message=(
                        "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
                        "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
                        "Please ensure the environment variables are set correctly and try again."
                    ),
                )
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

        error_files = list(Path(output_path).glob("error-*.txt"))
        if not error_files:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "No 'error-*.txt' files found in the output directory. There are two stages in the Grok run. "
                    "The profiling stage passed successfully, but something went wrong in the actual run stage. "
                    "Please ensure the actual run stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        for error_file in error_files:
            with open(error_file, "r") as file:
                content = file.read()
                if "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
                            "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
                            "Please ensure the environment variables are set correctly and try again."
                        ),
                    )
                if "E2E time: Elapsed time for" not in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"The file {error_file} does not contain the expected 'E2E time: Elapsed time for' keyword "
                            "at the end. This indicates the actual run did not complete successfully. "
                            "Please debug this manually to ensure the actual run stage completes as expected."
                        ),
                    )

        return JobStatusResult(is_successful=True)
