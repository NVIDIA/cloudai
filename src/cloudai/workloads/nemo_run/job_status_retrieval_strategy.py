# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import JobStatusResult, JobStatusRetrievalStrategy


class NeMoRunJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for NeMoRun by checking 'stderr.txt'."""

    def get_job_status(self, output_path: Path) -> JobStatusResult:
        stderr_path = output_path / "stderr.txt"
        if stderr_path.is_file():
            with stderr_path.open("r") as file:
                content = file.read()

                if "max_steps=" in content and "reached" in content:
                    return JobStatusResult(is_successful=True)

                missing_indicators = []
                if "max_steps=" not in content:
                    missing_indicators.append("'max_steps='")
                if "reached" not in content:
                    missing_indicators.append("'reached'")

                error_message = (
                    f"Missing success indicators in {stderr_path}: {', '.join(missing_indicators)}. "
                    "These keywords are expected to be present in stderr.txt when the NeMo training job "
                    "completes successfully. Please review the full stderr output. "
                    "Ensure that the NeMo training ran to completion and the logger output wasn't suppressed. "
                    "If the issue persists, contact the system administrator."
                )
                return JobStatusResult(is_successful=False, error_message=error_message)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stderr.txt file not found in the specified output directory {output_path}. "
                "This file is expected to be created as part of the NeMo training job. "
                "Please ensure the job was submitted and executed properly. "
                f"You can try re-running the job manually and verify that {stderr_path} is created "
                "with the expected output. If the issue persists, contact the system administrator."
            ),
        )
