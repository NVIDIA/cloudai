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

import logging
from pathlib import Path
from typing import cast

from cloudai import BaseJob, BaseRunner, Test
from cloudai.systems import KubernetesSystem

from .kubernetes_job import KubernetesJob


class KubernetesRunner(BaseRunner):
    """Implementation of the Runner for a system using Kubernetes."""

    def _submit_test(self, test: Test) -> KubernetesJob:
        """
        Submit a test for execution on Kubernetes and return a KubernetesJob object.

        Args:
            test (Test): The test to be executed.

        Returns:
            KubernetesJob: A KubernetesJob object containing job details.
        """
        logging.info(f"Running test: {test.section_name}")
        job_output_path = self.get_job_output_path(test)
        job_name = test.section_name.replace(".", "-").lower()
        job_spec = test.gen_json(job_output_path, job_name)
        job_kind = job_spec.get("kind", "").lower()
        logging.info(f"Generated JSON string for test {test.section_name}: {job_spec}")
        job_namespace = ""

        if self.mode == "run":
            k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
            job_name, job_namespace = k8s_system.create_job(job_spec)

        return KubernetesJob(self.mode, self.system, test, job_namespace, job_name, job_kind, job_output_path)

    async def store_job_logs(self, job: BaseJob) -> None:
        """
        Store logs for a Kubernetes job.

        Args:
            job (BaseJob): The Kubernetes job for which logs need to be stored.
        """
        k8s_system = cast(KubernetesSystem, self.system)
        k_job = cast(KubernetesJob, job)

        # Use Kubernetes API to list pods with the job's label selector
        label_selector = f"job-name={k_job.get_name()}"
        logging.info(f"Listing pods with label selector '{label_selector}' in namespace '{k_job.get_namespace()}'")

        pods = k8s_system.core_v1.list_namespaced_pod(
            namespace=k_job.get_namespace(), label_selector=label_selector
        ).items

        if not pods:
            logging.warning(f"No pods found for job '{k_job.get_name()}' in namespace '{k_job.get_namespace()}'")
        else:
            logging.info(f"Found {len(pods)} pod(s) for job '{k_job.get_name()}'")

        # Iterate over each pod and store its logs
        for pod in pods:
            pod_name = pod.metadata.name
            log_file_path = Path(k_job.output_path) / f"{pod_name}.txt"

            logging.info(f"Storing logs for pod '{pod_name}' to '{log_file_path}'")

            try:
                await k8s_system.store_pod_logs(k_job.get_namespace(), pod_name, log_file_path)
                logging.info(f"Logs for pod '{pod_name}' stored successfully at '{log_file_path}'")
            except Exception as e:
                logging.error(f"Failed to store logs for pod '{pod_name}': {e}")

    def kill_job(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated, casted to KubernetesJob.
        """
        k8s_system = cast(KubernetesSystem, self.system)
        k_job = cast(KubernetesJob, job)
        k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
        log_file_path = os.path.join(k_job.output_path, "stdout.txt")
        k8s_system.store_pod_logs(k_job.get_namespace(), k_job.get_name(), log_file_path)
        k8s_system.delete_job(k_job.get_name(), k_job.get_namespace())
