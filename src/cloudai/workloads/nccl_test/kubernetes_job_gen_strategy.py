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

from typing import Any, Dict, List, Union, cast

from cloudai._core.kubernetes_job_gen_strategy import JobSpec, JobStep, KubernetesJobGenStrategy
from cloudai._core.test_scenario import TestRun

from .nccl import NCCLTestDefinition


class NCCLTestKubernetesJobGenStrategy(KubernetesJobGenStrategy):
    """Job generation strategy for NCCL tests on Kubernetes systems."""

    SSH_PORT: int = 2222

    def generate_spec(self, tr: TestRun) -> JobSpec:
        manifest = self._generate_manifest(tr)
        return JobSpec(
            steps=[JobStep(name="submit_job", command_type="kubectl", args={"action": "apply", "manifest": manifest})],
            manifest=manifest,
        )

    def _generate_manifest(self, tr: TestRun) -> Dict[Any, Any]:
        return {
            "apiVersion": "kubeflow.org/v2beta1",
            "kind": "MPIJob",
            "metadata": {
                "name": self.sanitize_k8s_job_name("nccl-test"),
            },
            "spec": {
                "slotsPerWorker": 1,
                "runPolicy": {"cleanPodPolicy": "Running"},
                "mpiReplicaSpecs": {
                    "Launcher": self._create_launcher_spec(tr),
                    "Worker": self._create_worker_spec(tr),
                },
            },
        }

    def _create_launcher_spec(self, tr: TestRun) -> Dict[str, Any]:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        env_vars = self._get_merged_env_vars(tr)
        return {
            "replicas": 1,
            "template": {
                "spec": {
                    "hostNetwork": True,
                    "containers": [
                        {
                            "image": tdef.cmd_args.docker_image_url,
                            "name": "nccl-test-launcher",
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": {"privileged": True},
                            "env": self._generate_env_list(env_vars),
                            "command": ["/bin/bash", "-c"],
                            "args": [self._generate_launcher_command(tr, env_vars)],
                        }
                    ],
                },
            },
        }

    def _create_worker_spec(self, tr: TestRun) -> Dict[str, Any]:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        env_vars = self._get_merged_env_vars(tr)
        return {
            "replicas": tr.num_nodes,
            "template": {
                "spec": {
                    "hostNetwork": True,
                    "containers": [
                        {
                            "image": tdef.cmd_args.docker_image_url,
                            "name": "nccl-test-worker",
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": {"privileged": True},
                            "ports": [{"containerPort": self.SSH_PORT, "name": "ssh"}],
                            "env": self._generate_env_list(env_vars),
                            "command": ["/bin/bash"],
                            "args": ["-c", f"/usr/sbin/sshd -p {self.SSH_PORT}; sleep infinity"],
                            "resources": self._prepare_worker_resources(),
                            "volumeMounts": [
                                {"mountPath": "/dev/shm", "name": "dev-shm"},
                            ],
                        }
                    ],
                    "volumes": [{"name": "dev-shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}],
                },
            },
        }

    def _get_merged_env_vars(self, tr: TestRun) -> Dict[str, Union[str, List[str]]]:
        return self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)

    def _generate_env_list(self, env_vars: Dict[str, Union[str, List[str]]]) -> List[Dict[str, str]]:
        env_list = [{"name": "OMPI_ALLOW_RUN_AS_ROOT", "value": "1"}]
        for key, value in env_vars.items():
            if isinstance(value, list):
                value = ",".join(value)
            env_list.append({"name": key, "value": value})
        return env_list

    def _generate_mpi_args(self, env_vars: Dict[str, Union[str, List[str]]]) -> List[str]:
        mpi_args = [
            "--allow-run-as-root",
            f"--mca plm_rsh_args '-p {self.SSH_PORT}'",
            "-c 2",
            "-bind-to none -map-by slot",
            "-mca btl tcp,self",
        ]

        if "NCCL_SOCKET_IFNAME" in env_vars:
            mpi_args.append(f"-mca btl_tcp_if_include {env_vars['NCCL_SOCKET_IFNAME']}")

        return mpi_args

    def _generate_nccl_args(self, cmd_args_dict: Dict[str, Any]) -> List[str]:
        nccl_args = []
        for arg, value in cmd_args_dict.items():
            if value is not None:
                nccl_args.append(f"--{arg} {value}")
        return nccl_args

    def _generate_extra_args(self, extra_cmd_args: Dict[str, str]) -> List[str]:
        extra_args = []
        for key, value in extra_cmd_args.items():
            key = key if key.startswith("--") else f"--{key}"
            extra_args.append(f"{key} {value}" if value else key)
        return extra_args

    def _generate_launcher_command(self, tr: TestRun, env_vars: Dict[str, Union[str, List[str]]]) -> str:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        tdef_cmd_args = tdef.cmd_args

        cmd_args_dict = {
            k: v for k, v in tdef_cmd_args.model_dump().items() if k not in {"docker_image_url", "subtest_name"}
        }

        command_parts = [
            "mpirun",
            " ".join(self._generate_mpi_args(env_vars)),
            tdef_cmd_args.subtest_name,
            " ".join(self._generate_nccl_args(cmd_args_dict)),
        ]

        if tr.test.extra_cmd_args:
            command_parts.append(" ".join(self._generate_extra_args(tdef.extra_cmd_args)))

        return " \\\n".join(command_parts)

    def _prepare_launcher_resources(self) -> Dict[str, Dict[str, str]]:
        return {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "2", "memory": "8Gi"},
        }

    def _prepare_worker_resources(self) -> Dict[str, Dict[str, str]]:
        return {"requests": {"nvidia.com/gpu": "8"}, "limits": {"nvidia.com/gpu": "8"}}
