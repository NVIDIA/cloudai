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

from typing import Any, Dict, List

from cloudai import JsonGenStrategy


class NcclTestKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for NCCL tests on Kubernetes systems."""

    def gen_json(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        job_name: str,
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[Any, Any]:
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_num_nodes = self._determine_num_nodes(num_nodes, nodes)
        job_spec = self._create_job_spec(
            job_name, final_num_nodes, nodes, final_env_vars, final_cmd_args, extra_cmd_args
        )

        return job_spec

    def _determine_num_nodes(self, num_nodes: int, nodes: List[str]) -> int:
        """Determine the final number of nodes based on provided nodes or num_nodes."""
        return len(nodes) if nodes else num_nodes

    def _create_job_spec(
        self,
        job_name: str,
        final_num_nodes: int,
        nodes: List[str],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> Dict[Any, Any]:
        """Create the MPIJob specification for Kubernetes."""
        docker_image_url = "ghcr.io/coreweave/nccl-tests:12.4.1-cudnn-devel-ubuntu20.04-nccl2.21.5-1-85f9143"

        return {
            "apiVersion": "kubeflow.org/v2beta1",
            "kind": "MPIJob",
            "metadata": {
                "name": job_name,
            },
            "spec": {
                "slotsPerWorker": 1,
                "runPolicy": {"cleanPodPolicy": "Running"},
                "mpiReplicaSpecs": {
                    "Launcher": {
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "image": docker_image_url,
                                        "name": "nccl",
                                        "env": self._generate_env_list(env_vars),
                                        "command": ["/bin/bash", "-c"],
                                        "args": self._generate_command_args(
                                            final_num_nodes, nodes, cmd_args, extra_cmd_args
                                        ),
                                        "resources": self._prepare_launcher_resources(),
                                    }
                                ],
                                "restartPolicy": "Never",
                            }
                        },
                    },
                    "Worker": {
                        "replicas": final_num_nodes,
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "image": docker_image_url,
                                        "name": "nccl",
                                        "resources": self._prepare_resources(),
                                        "volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
                                    }
                                ],
                                "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
                            }
                        },
                    },
                },
            },
        }

    def _generate_env_list(self, env_vars: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate the environment variables list for the Kubernetes container."""
        return [
            {"name": "OMPI_ALLOW_RUN_AS_ROOT", "value": "1"},
            {"name": "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "value": "1"},
            {"name": "NCCL_DEBUG", "value": env_vars.get("NCCL_DEBUG", "INFO")},
            {"name": "NCCL_DEBUG_SUBSYS", "value": env_vars.get("NCCL_DEBUG_SUBSYS", "ALL")},
            {"name": "UCX_LOG_LEVEL", "value": env_vars.get("UCX_LOG_LEVEL", "debug")},
        ]

    def _generate_command_args(
        self, final_num_nodes: int, nodes: List[str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        """Generate the command arguments for the Kubernetes container."""
        subtest_name = cmd_args.get("subtest_name")
        if subtest_name is None:
            raise ValueError(
                "The NCCL test's 'subtest_name' is not provided. Please ensure 'subtest_name' "
                "is included in the command arguments. Valid subtest names include: "
                "all_reduce_perf, all_gather_perf, alltoall_perf, broadcast_perf, "
                "gather_perf, hypercube_perf, reduce_perf, reduce_scatter_perf, "
                "scatter_perf, and sendrecv_perf."
            )

        nccl_test_args = [
            "nthreads",
            "ngpus",
            "minbytes",
            "maxbytes",
            "stepbytes",
            "op",
            "datatype",
            "root",
            "iters",
            "warmup_iters",
            "agg_iters",
            "average",
            "parallel_init",
            "check",
            "blocking",
            "cudagraph",
        ]

        command_parts = [f"/opt/nccl_tests/build/{subtest_name}"]
        for arg in nccl_test_args:
            if arg in cmd_args:
                command_parts.append(f"--{arg} {cmd_args[arg]}")

        if extra_cmd_args:
            command_parts.append(extra_cmd_args)

        return [f"mpirun -np {final_num_nodes} -bind-to none {' '.join(command_parts)}"]

    def _prepare_launcher_resources(self) -> Dict[str, Dict[str, str]]:
        """Prepare resource requests and limits for the launcher container."""
        return {"requests": {"cpu": "2", "memory": "128Mi"}, "limits": {"cpu": "2", "memory": "128Mi"}}

    def _prepare_resources(self) -> Dict[str, Dict[str, str]]:
        """Prepare resource requests and limits for the worker containers."""
        return {
            "requests": {"cpu": "1", "memory": "32Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "1", "memory": "32Gi", "nvidia.com/gpu": "1"},
        }
