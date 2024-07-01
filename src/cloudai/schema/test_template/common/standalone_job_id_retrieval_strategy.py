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

from typing import Optional

from cloudai import JobIdRetrievalStrategy


class StandaloneJobIdRetrievalStrategy(JobIdRetrievalStrategy):
    """
    Strategy for retrieving job IDs from standalone application outputs.

    This class provides a specific implementation of JobIdRetrievalStrategy for extracting job IDs from the output of
    standalone applications, assuming the job ID is directly available in the standard output.
    """

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Extract the job ID from the standalone application's output.

        This simplistic approach assumes the entire stdout content is a job ID.

        Args:
            stdout (str): The standard output from the application.
            stderr (str): The standard error from the application (unused).

        Returns:
            Optional[int]: The extracted job ID as an integer, or None if the conversion fails due to invalid output.
        """
        try:
            return int(stdout)
        except ValueError:
            return None
