# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import yaml

from cloudai.core import JsonGenStrategy
from cloudai.systems.kubernetes import KubernetesSystem

from .nccl import NCCLTestDefinition


class NcclTestKubernetesJsonGenStrategy(JsonGenStrategy):
    """
    JSON generation strategy for NCCL tests on Kubernetes systems.

    This strategy generates an MPIJob configuration for running NCCL tests.
    """

    @property
    def ssh_port(self) -> int:
        return 2222

    def gen_json(self) -> dict[Any, Any]:
        k8s_system = cast(KubernetesSystem, self.system)
        job_name = self.sanitize_k8s_job_name(self.test_run.name)

        deployment = {
            "apiVersion": "kubeflow.org/v2beta1",
            "kind": "MPIJob",
            "metadata": {
                "name": job_name,
                "namespace": k8s_system.default_namespace,
            },
            "spec": {
                "slotsPerWorker": k8s_system.gpus_per_node,
                "runPolicy": {"cleanPodPolicy": "Running"},
                "mpiReplicaSpecs": {
                    "Launcher": self._create_launcher_spec(),
                    "Worker": self._create_worker_spec(),
                },
            },
        }

        with open(self.test_run.output_path / "deployment.yaml", "w") as f:
            yaml.dump(deployment, f)

        return deployment

    @property
    def container_url(self) -> str:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        return tdef.cmd_args.docker_image_url.replace("#", "/")

    def _create_launcher_spec(self) -> dict[str, Any]:
        env_vars = self._get_merged_env_vars()
        return {
            "replicas": 1,
            "template": {
                "spec": {
                    "hostNetwork": True,
                    "containers": [
                        {
                            "image": self.container_url,
                            "name": "nccl-test-launcher",
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": {"privileged": True},
                            "env": self._generate_env_list(env_vars),
                            "command": ["/bin/bash", "-c"],
                            "args": [self._generate_launcher_command()],
                            "resources": self._prepare_launcher_resources(),
                        }
                    ],
                },
            },
        }

    def _create_worker_spec(self) -> dict[str, Any]:
        env_vars = self._get_merged_env_vars()
        return {
            "replicas": self.test_run.nnodes,
            "template": {
                "spec": {
                    "hostNetwork": True,
                    "containers": [
                        {
                            "image": self.container_url,
                            "name": "nccl-test-worker",
                            "ports": [{"containerPort": self.ssh_port, "name": "ssh"}],
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": {"privileged": True},
                            "env": self._generate_env_list(env_vars),
                            "command": ["/bin/bash", "-c"],
                            "args": [self._generate_worker_command()],
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

    def _generate_worker_command(self) -> str:
        """
        Generate command for worker pods that starts the SSH daemon.

        If the SSH daemon is not installed, it will be installed and the SSH keys will be generated.
        """
        return f"""
set -ex
if ! command -v sshd &> /dev/null; then
    apt-get update && apt-get install -y --no-install-recommends openssh-server
fi
mkdir -p /var/run/sshd
cat >> /etc/ssh/sshd_config << EOF
PermitRootLogin yes
PubkeyAuthentication yes
StrictModes no
Port {self.ssh_port}
EOF
ssh-keygen -A
service ssh restart
sleep infinity
""".strip()

    def _get_merged_env_vars(self) -> dict[str, str | list[str]]:
        final_env_vars = self.system.global_env_vars.copy()
        final_env_vars.update(self.test_run.test.extra_env_vars)
        return final_env_vars

    def _generate_env_list(self, env_vars: Dict[str, Union[str, List[str]]]) -> List[Dict[str, str]]:
        env_list = [
            {"name": "OMPI_ALLOW_RUN_AS_ROOT", "value": "1"},
            {"name": "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "value": "1"},
        ]
        for key, value in env_vars.items():
            if isinstance(value, list):
                value = ",".join(value)
            env_list.append({"name": key, "value": value})
        return env_list

    def _generate_mpi_args(self) -> List[str]:
        k8s_system = cast(KubernetesSystem, self.system)
        total_processes = self.test_run.nnodes * k8s_system.gpus_per_node

        mpi_args = [
            f"-np {total_processes}",
            "-bind-to none",
            # Disable strict host key checking for SSH and ensure correct port is used
            f"-mca plm_rsh_args '-p {self.ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'",
        ]

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

    def _generate_launcher_command(self) -> str:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        tdef_cmd_args = tdef.cmd_args

        cmd_args_dict = {
            k: v for k, v in tdef_cmd_args.model_dump().items() if k not in {"docker_image_url", "subtest_name"}
        }

        command_parts = [
            "mpirun",
            " ".join(self._generate_mpi_args()),
            tdef_cmd_args.subtest_name,
            " ".join(self._generate_nccl_args(cmd_args_dict)),
        ]

        if self.test_run.test.extra_cmd_args:
            command_parts.append(" ".join(self._generate_extra_args(tdef.extra_cmd_args)))

        return " \\\n".join(command_parts)

    def _prepare_launcher_resources(self) -> Dict[str, Dict[str, str]]:
        return {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "2", "memory": "8Gi"},
        }

    def _prepare_worker_resources(self) -> Dict[str, Dict[str, str]]:
        k8s_system = cast(KubernetesSystem, self.system)
        gpu_count = str(k8s_system.gpus_per_node)
        return {"requests": {"nvidia.com/gpu": gpu_count}, "limits": {"nvidia.com/gpu": gpu_count}}
