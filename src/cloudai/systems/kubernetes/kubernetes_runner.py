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

from .kubernetes_job import KubernetesJob
from .kubernetes_system import KubernetesSystem


class KubernetesRunner(BaseRunner):
    """Implementation of the Runner for a system using Kubernetes."""

    def _submit_test(self, tr: TestRun) -> KubernetesJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        job_name = tr.name.replace(".", "-").lower()
        job_spec = self.get_json_gen_strategy(self.system, tr).gen_json()
        job_kind = job_spec.get("kind", "").lower()
        logging.debug(f"Generated JSON string for test {tr.name}: {job_spec}")

        if self.mode == "run":
            k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
            job_name = k8s_system.create_job(job_spec)

        job = KubernetesJob(tr, id=job_name, name=job_name, kind=job_kind)

        return job

    def on_job_submit(self, tr: TestRun) -> None:
        json_gen = self.get_json_gen_strategy(self.system, tr)
        json_gen.store_test_run()

    def on_job_completion(self, job: BaseJob) -> None:
        k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
        k_job = cast(KubernetesJob, job)
        if k_job.kind == "dynamographdeployment":
            k8s_system._delete_dynamo_graph_deployment(k_job.name)
        else:
            k8s_system.store_logs_for_job(k_job.name, k_job.test_run.output_path)
            k8s_system.delete_job(k_job.name, k_job.kind)

    def kill_job(self, job: BaseJob) -> None:
        k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
        k_job = cast(KubernetesJob, job)
        k8s_system.store_logs_for_job(k_job.name, k_job.test_run.output_path)
        k8s_system.delete_job(k_job.name, k_job.kind)
