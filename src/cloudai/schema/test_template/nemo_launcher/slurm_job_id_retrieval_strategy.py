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

import re
from typing import Optional

from cloudai import JobIdRetrievalStrategy


class NeMoLauncherSlurmJobIdRetrievalStrategy(JobIdRetrievalStrategy):
    """
    Strategy for retrieving job IDs from NeMo launcher submissions to a Slurm scheduler.

    This class implements the JobIdRetrievalStrategy interface to extract job IDs from the standard output provided by
    the Slurm scheduler when the NeMo launcher template is used for job submission.
    """

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Extract the job ID from the Slurm command output.

        This method searches the standard output for a specific pattern that matches the submission confirmation
        message of a Slurm job and extracts the job ID.

        Args:
            stdout (str): The standard output from the Slurm command.
            stderr (str): The standard error from the Slurm command (unused).

        Returns:
            Optional[int]: The extracted job ID if found, otherwise None.
        """
        match = re.search(r"submitted with Job ID (\d+)", stdout)

        if match:
            return int(match.group(1))
        else:
            return None
