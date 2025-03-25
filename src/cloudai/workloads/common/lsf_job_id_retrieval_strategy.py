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

import re
from typing import Optional

from cloudai import JobIdRetrievalStrategy


class LSFJobIdRetrievalStrategy(JobIdRetrievalStrategy):
    """
    Strategy for retrieving job IDs from LSF job submission outputs.

    Implements JobIdRetrievalStrategy to extract job IDs from the standard output of LSF scheduler submissions.
    """

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Extract the job ID from LSF command output.

        Parses stdout for a pattern matching LSF submission confirmation and extracts the job ID.

        Args:
            stdout (str): Standard output from the LSF command.
            stderr (str): Standard error from the LSF command.

        Returns:
            Optional[int]: Extracted job ID, or None if not found.
        """
        match = re.search(r"Job <(\d+)> is submitted", stdout)
        if match:
            return int(match.group(1))
        return None
