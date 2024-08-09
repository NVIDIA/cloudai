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
        mode (str): The mode of the job (e.g., 'run', 'dry-run').
        system (System): The system in which the job is running.
        test (Test): The test instance associated with this job.
        id (Union[str, int]): The unique identifier of the job.
        output_path (Path): The path where the job's output is stored.
    """

    def __init__(self, mode: str, system: System, test: Test, id: Union[str, int], output_path: Path):
        BaseJob.__init__(self, mode, system, test, output_path)
        self.id = id

    def get_id(self) -> Union[str, int]:
        """
        Retrieve the unique identifier of the job.

        Returns
            Union[str, int]: The unique identifier of the job.
        """
        return self.id

    def __repr__(self) -> str:
        """
        Return a string representation of the StandaloneJob instance.

        Returns
            str: String representation of the job.
        """
        return f"StandaloneJob(id={self.get_id()}, test={self.test.name})"
