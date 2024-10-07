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


from cloudai import BaseJob, System, TestRun


class KubernetesJob(BaseJob):
    """
    A job class for execution on a Kubernetes system.

    Attributes
        mode (str): The mode of the job (e.g., 'run', 'dry-run').
        system (System): The system in which the job is running.
        test_run (TestRun): The test instance associated with this job.
        namespace (str): The namespace of the job.
        name (str): The name of the job.
        kind (str): The kind of the job.
    """

    def __init__(self, mode: str, system: System, test_run: TestRun, namespace: str, name: str, kind: str):
        """
        Initialize a KubernetesJob instance.

        Args:
            mode (str): The mode of the job (e.g., 'run', 'dry-run').
            system (System): The system in which the job is running.
            test_run (TestRun): The test instance associated with this job.
            namespace (str): The namespace of the job.
            name (str): The name of the job.
            kind (str): The kind of the job.
        """
        super().__init__(mode, system, test_run)
        self.id = name
        self.namespace = namespace
        self.name = name
        self.kind = kind

    def __repr__(self) -> str:
        """
        Return a string representation of the KubernetesJob instance.

        Returns
            str: String representation of the job.
        """
        return (
            f"KubernetesJob(name={self.name}, namespace={self.namespace}, test={self.test_run.test.name}, "
            f"kind={self.kind})"
        )
