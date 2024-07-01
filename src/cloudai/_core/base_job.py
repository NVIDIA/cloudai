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

from .test import Test


class BaseJob:
    """
    Base class for representing a job created by executing a test.

    Attributes
        id (int): The unique identifier of the job.
        test (Test): The test instance associated with this job.
        output_path (str): The path where the job's output is stored.
        terminated_by_dependency (bool): Flag to indicate if the job was terminated due to a dependency.
    """

    def __init__(self, job_id: int, test: Test, output_path: str):
        """
        Initialize a BaseJob instance.

        Args:
            job_id (int): The unique identifier of the job.
            output_path (str): The path where the job's output is stored.
            test (Test): The test instance associated with the job.
        """
        self.id = job_id
        self.test = test
        self.output_path = output_path
        self.terminated_by_dependency = False

    def increment_iteration(self):
        """
        Increment the iteration count of the associated test.

        This method should be called when the job completes an iteration and is ready to proceed to the next one.
        """
        self.test.current_iteration += 1

    def __repr__(self) -> str:
        """
        Return a string representation of the BaseJob instance.

        Returns
            str: String representation of the job.
        """
        return f"BaseJob(id={self.id}, test={self.test.name}, terminated_by_dependency={self.terminated_by_dependency})"
