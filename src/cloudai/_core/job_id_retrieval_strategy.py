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

from abc import abstractmethod
from typing import Optional


class JobIdRetrievalStrategy:
    """
    Abstract class to define a strategy for retrieving job IDs from given stdout and stderr streams.

    Attributes
        None

    Methods
        get_job_id(stdout: str, stderr: str) -> Optional[int]:
            Abstract method to be implemented by subclasses for extracting a job ID.
    """

    @abstractmethod
    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Retrieve the job ID from stdout and stderr outputs.

        Args:
            stdout (str): The standard output stream from a job submission.
            stderr (str): The standard error stream from a job submission.

        Returns:
            Optional[int]: The extracted job ID if found, otherwise None.
        """
        pass
