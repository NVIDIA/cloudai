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

from pathlib import Path
from typing import Union

from cloudai import BaseJob, System, Test


class StandaloneJob(BaseJob):
    """
    A job class for standalone execution.

    Attributes
        id (Union[str, int]): The unique identifier of the job.
    """

    def __init__(self, mode: str, system: System, test: Test, job_id: Union[str, int], output_path: Path):
        BaseJob.__init__(self, mode, system, test, output_path)
        self.id = job_id

    def __repr__(self) -> str:
        """
        Return a string representation of the StandaloneJob instance.

        Returns
            str: String representation of the job.
        """
        return f"StandaloneJob(id={self.id}, test={self.test.name})"
