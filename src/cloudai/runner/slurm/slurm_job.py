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

from cloudai import BaseJob, System, TestRun


class SlurmJob(BaseJob):
    """
    A job class for execution on a Slurm system.

    Attributes
        id (Union[str, int]): The unique identifier of the job.
    """

    def __init__(self, mode: str, system: System, test_run: TestRun, job_id: Union[str, int]):
        BaseJob.__init__(self, mode, system, test_run)
        self.id = job_id

    def __repr__(self) -> str:
        """
        Return a string representation of the SlurmJob instance.

        Returns
            str: String representation of the job.
        """
        return f"SlurmJob(id={self.id}, test={self.test_run.test.name})"
