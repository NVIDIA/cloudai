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
import subprocess
from pathlib import Path
from typing import cast

from cloudai.core import BaseJob, BaseRunner, TestRun
from cloudai.workloads.ai_dynamo.ai_dynamo import AIDynamoTestDefinition

from .kubernetes_job import KubernetesJob
from .kubernetes_system import KubernetesSystem


class KubernetesRunner(BaseRunner):
    """Implementation of the Runner for a system using Kubernetes."""

    def _submit_test(self, tr: TestRun) -> KubernetesJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        job_name = tr.name.replace(".", "-").lower()
        job_spec = tr.test.test_template.gen_json(tr)
        job_kind = job_spec.get("kind", "").lower()
        logging.info(f"Generated JSON string for test {tr.name}: {job_spec}")

        if self.mode == "run":
            k8s_system: KubernetesSystem = cast(KubernetesSystem, self.system)
            job_name = k8s_system.create_job(job_spec)

        job = KubernetesJob(tr, id=job_name, name=job_name, kind=job_kind)

        if job_kind == "dynamographdeployment":
            self._setup_dynamo_graph_deployment(job, tr)

        return job

    def _setup_dynamo_graph_deployment(self, job: KubernetesJob, tr: TestRun) -> None:
        test_definition = tr.test.test_definition
        if not isinstance(test_definition, AIDynamoTestDefinition):
            raise TypeError("Test definition must be an instance of AIDynamoTestDefinition")

        python_exec = test_definition.python_executable
        if not python_exec.venv_path:
            raise ValueError(
                f"The virtual environment for git repo {python_exec.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )

        venv_pip = python_exec.venv_path.absolute() / "bin" / "pip"
        assert python_exec.git_repo.installed_path
        repo_root = python_exec.git_repo.installed_path.absolute()

        self._install_python_packages(repo_root, venv_pip)

        job.python_executable = python_exec
        job.genai_perf_args = test_definition.cmd_args.genai_perf
        job.output_path = tr.output_path

    def _install_python_packages(self, repo_root: Path, venv_pip: Path) -> None:
        installs = [
            ("perf_analyzer", repo_root),
            ("genai-perf", repo_root / "genai-perf"),
        ]

        for package, path in installs:
            install_cmd = f"cd {path} && {venv_pip} install ."
            logging.info(f"Installing {package} with command: {install_cmd}")
            subprocess.run(install_cmd, shell=True, capture_output=True, text=True, check=True)

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
