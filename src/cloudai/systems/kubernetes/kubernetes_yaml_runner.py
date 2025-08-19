# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import BaseJob, BaseRunner, TestRun

from .kubernetes_yaml_job import KubernetesYAMLJob
from .kubernetes_yaml_system import KubernetesYAMLSystem


class KubernetesYAMLRunner(BaseRunner):
    """Implementation of the Runner for Kubernetes YAML-based deployments."""

    def _submit_test(self, tr: TestRun) -> KubernetesYAMLJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        job_name = tr.name.replace(".", "-").lower()
        job_spec = tr.test.test_template.gen_json(tr)
        job_kind = job_spec.get("kind", "").lower()
        logging.info(f"Generated JSON string for test {tr.name}: {job_spec}")

        if self.mode == "run":
            k8s_system: KubernetesYAMLSystem = cast(KubernetesYAMLSystem, self.system)
            job_name = k8s_system.create_job(job_spec)

        return KubernetesYAMLJob(tr, id=job_name, name=job_name, kind=job_kind)

    def on_job_completion(self, job: BaseJob) -> None:
        k8s_system: KubernetesYAMLSystem = cast(KubernetesYAMLSystem, self.system)
        k_job = cast(KubernetesYAMLJob, job)
        k8s_system.store_logs_for_job(k_job.name, k_job.test_run.output_path)
        k8s_system.delete_job(k_job.name, k_job.kind)

    def kill_job(self, job: BaseJob) -> None:
        k8s_system: KubernetesYAMLSystem = cast(KubernetesYAMLSystem, self.system)
        k_job = cast(KubernetesYAMLJob, job)
        k8s_system.store_logs_for_job(k_job.name, k_job.test_run.output_path)
        k8s_system.delete_job(k_job.name, k_job.kind)
