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

from cloudai import BaseJob
from cloudai._core.test import Test

class KubernetesJob(BaseJob):
    """
    A class for representing a kubernetes job created by executing a test.

    Attributes
        name (str): The name of the job.
        namespace (str): The namespace of the job.
        test (Test): The test instance associated with this job.
    """

    def __init__(self, job_id: int, test: Test, output_path: str, name: str, namespace: str):
        """
        Initialize a Kubernetes Job instance.

        Args:
            name (str): The name of the job.
            namespace (str): The namespace of the job.
            test (Test): The test instance associated with this job.
        """
        super().__init__(job_id, test, output_path)
        self.namespace = namespace
        self.name = name

    def __repr__(self) -> str:
        """
        Return a string representation of the kubernetes Job instance.

        Returns
            str: String representation of the kubernetes job.
        """
        return f"BaseJob(name={self.name}, namespace={self.namespace}, test={self.test.name}, )"
