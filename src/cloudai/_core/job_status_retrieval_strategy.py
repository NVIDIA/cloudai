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
from typing import TYPE_CHECKING

from .job_status_result import JobStatusResult

if TYPE_CHECKING:
    from .base_job import BaseJob
    from .system import System


class JobStatusRetrievalStrategy:
    """Abstract base class for retrieving job statuses from a system."""

    @abstractmethod
    def get_job_status(self, system: "System", job: "BaseJob") -> JobStatusResult:
        """
        Retrieve the status of the specified job from the system.

        Args:
            system (System): The system where the job is running.
            job (BaseJob): The job to check.

        Returns:
            JobStatusResult: The status of the job, including any error messages.
        """
        pass
