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
from typing import cast

from cloudai import BaseJob, BaseRunner, TestRun
from cloudai.systems import KubernetesSystem

from .kubernetes_job import KubernetesJob


class KubernetesRunner(BaseRunner):
    """Implementation of the Runner for a system using Kubernetes."""

    def _submit_test(self, tr: TestRun) -> KubernetesJob:
        """
        Submit a test for execution on Kubernetes and return a KubernetesJob object.

        Args:
            tr (TestRun): The test run to be executed.

        Returns:
            KubernetesJob: A KubernetesJob object containing job details.
        """
        logging.info(f"Running test: {tr.test.section_name}")
        job_output_path = self.get_job_output_path(tr.test)
        job_name = tr.test.section_name.replace(".", "-").lower()
        job_spec = tr.test.gen_json(job_output_path, job_name)
        job_kind = job_spec.get("kind", "").lower()
        logging.info(f"Generated JSON string for test {tr.test.section_name}: {job_spec}")
        job_namespace = ""

        if self.mode == "run":
            k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
            job_name, job_namespace = k8s_system.create_job(job_spec)

        return KubernetesJob(self.mode, self.system, tr, job_namespace, job_name, job_kind, job_output_path)

    def kill_job(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated, casted to KubernetesJob.
        """
        k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
        k_job = cast(KubernetesJob, job)
        k8s_system.delete_job(k_job.namespace, k_job.name, k_job.kind)
