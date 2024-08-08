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

from cloudai import BaseJob, System, Test


class KubernetesJob(BaseJob):
    """
    A job class for execution on a Kubernetes system.

    Attributes
        mode (str): The mode of the job (e.g., 'run', 'dry-run').
        system (System): The system in which the job is running.
        test (Test): The test instance associated with this job.
        namespace (str): The namespace of the job.
        name (str): The name of the job.
        kind (str): The kind of the job.
        output_path (str): The path where the job's output is stored.
    """

    def __init__(self, mode: str, system: System, test: Test, namespace: str, name: str, kind: str, output_path: str):
        BaseJob.__init__(self, mode, system, test, output_path)
        self.namespace = namespace
        self.name = name
        self.kind = kind

    def get_id(self) -> Union[str, int]:
        """
        Retrieve the unique name of the Kubernetes job.

        Returns
            Union[str, int]: The unique identifier of the job.
        """
        return self.name

    def get_namespace(self) -> str:
        """
        Retrieve the namespace of the Kubernetes job.

        Returns
            str: The namespace of the job.
        """
        return self.namespace

    def get_name(self) -> str:
        """
        Retrieve the name of the Kubernetes job.

        Returns
            str: The name of the job.
        """
        return self.name

    def get_kind(self) -> str:
        """
        Retrieve the kind of the Kubernetes job.

        Returns
            str: The kind of the job.
        """
        return self.kind

    def __repr__(self) -> str:
        """
        Return a string representation of the KubernetesJob instance.

        Returns
            str: String representation of the job.
        """
        return f"KubernetesJob(name={self.name}, namespace={self.namespace}, test={self.test.name}, kind={self.kind})"
