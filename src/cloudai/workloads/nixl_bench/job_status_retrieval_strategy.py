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


class NIXLBenchJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for NIXL Bench by checking 'stdout.txt'."""

    def get_job_status(self, output_path: Path) -> JobStatusResult:
        stdout_path = output_path / "stdout.txt"
        if not stdout_path.exists():
            return JobStatusResult(
                is_successful=False,
                error_message=f"stdout.txt file not found in the specified output directory {output_path}.",
            )

        has_header, has_data = False, False
        for line in stdout_path.read_text().splitlines():
            if "Block Size (B)      Batch Size     Avg Lat. (us)  B/W (MiB/Sec)  B/W (GiB/Sec)  B/W (GB/Sec)" in line:
                has_header = True
                continue
            if has_header and len(line.split()) == 6:
                has_data = True
                break

        if has_data:
            return JobStatusResult(is_successful=True)

        if not has_header:
            return JobStatusResult(
                is_successful=False,
                error_message=f"NIXLBench results table not found in {stdout_path}.",
            )

        return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {stdout_path}.")
