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

from typing import Any, Dict, List, Union

from cloudai import JsonGenStrategy, TestRun


class NcclTestKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for NCCL tests on Kubernetes systems."""

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        final_env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        final_num_nodes = self._determine_num_nodes(tr.num_nodes, tr.nodes)
        sanitized_job_name = self.sanitize_k8s_job_name("nccl-test")
        job_spec = self._create_job_spec(
            sanitized_job_name, final_num_nodes, tr.nodes, final_env_vars, final_cmd_args, tr.test.extra_cmd_args
        )

        return job_spec

    def _determine_num_nodes(self, num_nodes: int, nodes: List[str]) -> int:
        """
        Determine the final number of nodes based on provided nodes or num_nodes.

        Args:
            num_nodes (int): The initial number of nodes specified.
            nodes (List[str]): A list of specific nodes provided.

        Returns:
            int: The final number of nodes to be used for the job.
        """
        return len(nodes) if nodes else num_nodes

    def _create_job_spec(
        self,
        job_name: str,
        final_num_nodes: int,
        nodes: List[str],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Union[str, List[str]]],
        extra_cmd_args: str,
    ) -> Dict[Any, Any]:
        """
        Create the MPIJob specification for running NCCL tests on a Kubernetes cluster.

        Args:
            job_name (str): The name of the Kubernetes job.
            final_num_nodes (int): The final number of nodes determined.
            nodes (List[str]): A list of specific nodes to run the job on.
            env_vars (Dict[str, str]): A dictionary of environment variables for the job.
            cmd_args (Dict[str, str]): A dictionary of command-line arguments for the NCCL test.
            extra_cmd_args (str): Additional command-line arguments for the NCCL test.

        Returns:
            Dict[Any, Any]: A dictionary representing the Kubernetes MPIJob specification.
        """
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
                                        "image": cmd_args["docker_image_url"],
                                        "name": "nccl-launcher",
                                        "env": self._generate_env_list(env_vars),
                                        "command": ["/bin/bash"],
                                        "args": [
                                            "-c",
                                            self._generate_launcher_command(
                                                final_num_nodes, nodes, env_vars, cmd_args, extra_cmd_args
                                            ),
                                        ],
                                        "resources": self._prepare_launcher_resources(),
                                    }
                                ],
                                "restartPolicy": "Never",
                            },
                        },
                    },
                    "Worker": {
                        "replicas": final_num_nodes,
                        "template": {
                            "spec": {
                                "hostNetwork": True,
                                "containers": [
                                    {
                                        "image": cmd_args["docker_image_url"],
                                        "name": "nccl-worker",
                                        "env": self._generate_env_list(env_vars),
                                        "command": ["/bin/bash"],
                                        "args": ["-c", "/usr/sbin/sshd -p 2222; sleep infinity"],
                                        "resources": self._prepare_worker_resources(),
                                        "volumeMounts": [
                                            {"mountPath": "/dev/shm", "name": "dshm"},
                                        ],
                                    }
                                ],
                                "volumes": [
                                    {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                                    {"name": "hugepage-2mi", "emptyDir": {"medium": "HugePages"}},
                                ],
                            },
                        },
                    },
                },
            },
        }

    def _generate_env_list(self, env_vars: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Generate the environment variables list for the Kubernetes container.

        Args:
            env_vars (Dict[str, str]): A dictionary of environment variables to include.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the environment variables.
        """
        env_list = [{"name": "OMPI_ALLOW_RUN_AS_ROOT", "value": "1"}]
        # Include additional environment variables from env_vars
        for key, value in env_vars.items():
            env_list.append({"name": key, "value": value})
        return env_list

    def _generate_launcher_command(
        self,
        final_num_nodes: int,
        nodes: List[str],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Union[str, List[str]]],
        extra_cmd_args: str,
    ) -> str:
        """
        Generate the launcher command for the Kubernetes container.

        Args:
            final_num_nodes (int): The final number of nodes determined.
            nodes (List[str]): A list of specific nodes to run the job on.
            env_vars (Dict[str, str]): A dictionary of environment variables for the job.
            cmd_args (Dict[str, str]): A dictionary of command-line arguments for the NCCL test.
            extra_cmd_args (str): Additional command-line arguments for the NCCL test.

        Returns:
            str: The launcher command to be executed.
        """
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

        return (
            f"mpirun -v --allow-run-as-root -np {final_num_nodes} "
            "--hostfile /etc/mpi/hostfile "
            "-mca coll ^hcoll -mca plm_rsh_args '-p 2222' -bind-to none "
            f"{' '.join([f'-x {key}={value}' for key, value in env_vars.items()])} "
            f"{' '.join(command_parts)}"
        )

    def _prepare_launcher_resources(self) -> Dict[str, Dict[str, str]]:
        """
        Prepare resource requests and limits for the launcher container.

        Returns
            Dict[str, Dict[str, str]]: A dictionary representing the resource requests and limits.
        """
        return {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "2", "memory": "8Gi"},
        }

    def _prepare_worker_resources(self) -> Dict[str, Dict[str, str]]:
        """
        Prepare resource requests and limits for the worker containers.

        Returns
            Dict[str, Dict[str, str]]: A dictionary representing the resource requests and limits.
        """
        return {
            "requests": {
                "cpu": "24",
                "memory": "32Gi",
                "nvidia.com/gpu": "1",
                "rdma/rdma_ib": "1",
                "hugepages-2Mi": "2Gi",
            },
            "limits": {
                "cpu": "48",
                "memory": "32Gi",
                "nvidia.com/gpu": "1",
                "rdma/rdma_ib": "1",
                "hugepages-2Mi": "2Gi",
            },
        }
