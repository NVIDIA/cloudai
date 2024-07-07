#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from typing import Optional

from cloudai import JobIdRetrievalStrategy


class SlurmJobIdRetrievalStrategy(JobIdRetrievalStrategy):
    """
    Strategy for retrieving job IDs from Slurm job submission outputs.

    Implements JobIdRetrievalStrategy to extract job IDs from the standard output of Slurm scheduler submissions.
    """

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Extract the job ID from Slurm command output.

        Parses stdout for a pattern matching Slurm submission confirmation and extracts the job ID.

        Args:
            stdout (str): Standard output from the Slurm command.
            stderr (str): Standard error from the Slurm command.

        Returns:
            Optional[int]: Extracted job ID, or None if not found.
        """
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            return int(match.group(1))
        return None
