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

import os
from time import sleep
from typing import List
from kubernetes import config, client

class KubernetesJobClient:
    """
    A class responsible for submit job to kubernetes cluster.

    Attributes
        kube_config_path (str): The kube config file path.
    """

    def __init__(self, kube_config_path: str):
        """
        Initialize the KubernetesJobClient.

        Args:
            kube_config_path (str): The kube config file path. Defaults to "~/.kube/config".

        Raises:
            FileNotFoundError: If the specified kube config file does not exist.
        """
        if not os.path.exists(kube_config_path):
            raise FileNotFoundError(f"kube config file '{kube_config_path}' not found.")
        self.kube_config_path = kube_config_path
        config.load_kube_config(config_file = self.kube_config_path)
        self.core_v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()

    def create_node_group(self, name: str, node_list: List[str]):
        for node in node_list:
            body = {
                "metadata": {
                    "labels": { "cloudai/node-group": name }
                }
            }
            self.core_v1.patch_node(node, body)

    """
    def create_job_object(self, job_name: str, image: str, args: List[str]):
        containers = client.V1Container(
            name = "task",
            image = image,
            command = ["/bin/bash", "-c"],
            args = args)
        # Create and configure a spec section
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(),
            spec=client.V1PodSpec(restart_policy = "Never", containers = [containers]))
        # Create the specification of deployment
        spec = client.V1JobSpec(
            template = template,
            backoff_limit = 4)
        # Instantiate the job object
        job = client.V1Job(
            api_version = "batch/v1",
            kind = "Job",
            metadata = client.V1ObjectMeta(name=job_name),
            spec = spec)

        return job
    """

    def create_job(self, job_spec, namespace) -> tuple[str, str]:
        api_response = self.batch_v1.create_namespaced_job(
            body = job_spec,
            namespace = namespace)
        print(f"Job created. status='{str(api_response.status)}'")
        return api_response.metadata.name, api_response.metadata.namespace

    def is_job_running(self, job_name, namespace) -> bool:
        k8s_job = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)
        if k8s_job.status.completion_time:
            return False
        else:
            return True

    def is_job_completed(self, job_name, namespace) -> bool:
        return not self.is_job_running(job_name, namespace)

    def get_job_status(self, job_name, namespace):
        job_completed = False
        while not job_completed:
            api_response = self.batch_v1.read_namespaced_job_status(
                name = job_name,
                namespace = namespace)
            if api_response.status.succeeded is not None or \
                    api_response.status.failed is not None:
                job_completed = True
            sleep(1)
            print(f"Job status='{str(api_response.status)}'")

    def delete_job(self, job_name, namespace):
        api_response = self.batch_v1.delete_namespaced_job(
            name = job_name,
            namespace = namespace,
            body = client.V1DeleteOptions(
                propagation_policy = 'Foreground',
                grace_period_seconds = 5))
        print(f"Job deleted. status='{str(api_response.status)}'")

    def list_job(self, namespace):
        return self.batch_v1.list_namespaced_job(namespace = namespace)


