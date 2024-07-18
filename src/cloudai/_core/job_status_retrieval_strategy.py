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

from .job_status_result import JobStatusResult


class JobStatusRetrievalStrategy:
    """Abstract class to define a strategy for retrieving job statuses from a given output directory."""

    @abstractmethod
    def get_job_status(self, output_path: str) -> JobStatusResult:
        """
        Retrieve the job status from a specified output directory.

        Args:
            output_path (str): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        pass
