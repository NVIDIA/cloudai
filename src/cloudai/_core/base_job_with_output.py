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

from typing import Union

from .base_job_with_id import BaseJobWithID
from .system import System
from .test import Test


class BaseJobWithOutput(BaseJobWithID):
    """
    Base class for representing a job with a unique identifier and an output path.

    Attributes
        id (Union[str, int]): The unique identifier of the job.
        output_path (str): The path where the job's output is stored.
    """

    def __init__(self, mode: str, system: System, test: Test, job_id: Union[str, int], output_path: str):
        """
        Initialize a BaseJobWithOutput instance.

        Args:
            mode (str): The mode of the job (e.g., 'run', 'dry-run').
            system (System): The system in which the job is running.
            test (Test): The test instance associated with the job.
            job_id (Union[str, int]): The unique identifier of the job.
            output_path (str): The path where the job's output is stored.
        """
        super().__init__(mode, system, test, job_id)
        self.output_path = output_path
