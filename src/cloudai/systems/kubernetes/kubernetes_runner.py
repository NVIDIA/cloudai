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

import asyncio
import uuid
from pathlib import Path
from typing import List, cast

import aiohttp

from cloudai._core.kubernetes_job_gen_strategy import JobStep, KubernetesJobGenStrategy
from cloudai.core import BaseRunner, Registry, TestRun

from .kubernetes_job import KubernetesJob
from .kubernetes_system import KubernetesSystem


class StepExecutionError(Exception):
    """Exception raised when a job step execution fails."""

    def __init__(self, step: JobStep, message: str, cause: Exception | None = None):
        self.step = step
        self.message = message
        self.cause = cause
        super().__init__(f"Step '{step.name}' failed: {message}")


class KubernetesRunner(BaseRunner):
    """Implementation of the Runner for a system using Kubernetes."""

    async def _submit_test(self, tr: TestRun) -> KubernetesJob:
        strategy = Registry().get_strategy(KubernetesJobGenStrategy, self.system.__class__, tr.test.__class__)
        job_spec = strategy.generate_spec(tr)

        job_id = f"{tr.name}-{uuid.uuid4().hex[:8]}"
        work_dir = self._prepare_work_dir(job_id)

        # Store task reference for proper lifecycle management
        self._current_task = asyncio.create_task(self._execute_steps(job_id, job_spec.steps, work_dir))

        return KubernetesJob(
            test_run=tr,
            name=job_id,
            kind="step-based" if not job_spec.manifest else job_spec.manifest["kind"],
            metadata={"spec": job_spec},
        )

    def _prepare_work_dir(self, job_id: str) -> Path:
        work_dir = self.system.output_path / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    async def _execute_steps(self, job_id: str, steps: List[JobStep], work_dir: Path):
        executed, failed = set(), set()

        while len(executed) + len(failed) < len(steps):
            for step in steps:
                if step.name in executed or step.name in failed:
                    continue

                if step.depends_on and not all(dep in executed for dep in step.depends_on):
                    continue

                try:
                    await self._execute_step(step, work_dir)
                    executed.add(step.name)
                except Exception as e:
                    failed.add(step.name)
                    if step.required:
                        raise StepExecutionError(step, str(e), e) from e

    async def _execute_step(self, step: JobStep, work_dir: Path):
        step_executors = {
            "helm": self._execute_helm_step,
            "kubectl": self._execute_kubectl_step,
            "port_forward": self._execute_port_forward_step,
            "http": self._execute_http_step,
        }

        executor = step_executors.get(step.command_type)
        if not executor:
            raise ValueError(f"Unsupported step type: {step.command_type}")

        await executor(step)

    async def _execute_helm_step(self, step: JobStep):
        k8s_system = cast(KubernetesSystem, self.system)
        args = step.args
        await k8s_system.helm_upgrade(
            release=args["release"],
            chart=args["chart_path"],
            namespace=args.get("namespace", "default"),
            values=args.get("values", {}),
        )

    async def _execute_kubectl_step(self, step: JobStep):
        k8s_system = cast(KubernetesSystem, self.system)
        args = step.args
        if args["action"] == "apply":
            if "manifest" in args:
                await k8s_system.apply_manifest(args["manifest"])
            elif "file" in args:
                await k8s_system.apply_file(args["file"])
        elif args["action"] == "wait":
            await k8s_system.wait_for_resource(
                resource=args["resource"],
                selector=args["selector"],
                condition=args["condition"],
                namespace=args.get("namespace"),
            )

    async def _execute_port_forward_step(self, step: JobStep):
        k8s_system = cast(KubernetesSystem, self.system)
        args = step.args
        await k8s_system.setup_port_forward(
            pod_selector=args["pod_selector"],
            local_port=args["local_port"],
            remote_port=args["remote_port"],
            namespace=args.get("namespace"),
        )

    async def _execute_http_step(self, step: JobStep):
        args = step.args
        async with (
            aiohttp.ClientSession() as session,
            session.request(
                method=args["method"], url=args["url"], headers=args.get("headers", {}), json=args.get("body")
            ) as response,
        ):
            if not response.ok:
                raise RuntimeError(f"HTTP request failed: {response.status}")

    def is_job_running(self, job: KubernetesJob) -> bool:
        if job.is_step_based:
            # Check if background task is still running
            return self._is_step_execution_running(job)
        else:
            # Use system's job status check for simple jobs
            return self.system.is_job_running(job.name, job.kind)

    def on_job_completion(self, job: KubernetesJob) -> None:
        if job.is_step_based:
            self._cleanup_step_based_job(job)
        else:
            # Use system's cleanup for simple jobs
            self.system.store_logs_for_job(job.name, job.test_run.output_path)
            self.system.delete_job(job.name, job.kind)

    def kill_job(self, job: KubernetesJob) -> None:
        if job.is_step_based:
            self._kill_step_based_job(job)
        else:
            self.system.delete_job(job.name, job.kind)
