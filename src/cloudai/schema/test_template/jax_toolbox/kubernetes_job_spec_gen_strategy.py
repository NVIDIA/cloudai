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

from pathlib import Path
from typing import Any, Dict, List, cast

from cloudai.systems import KubernetesSystem

from .base_job_spec_gen_strategy import JaxToolboxBaseJobSpecGenStrategy


class JaxToolboxKubernetesJobSpecGenStrategy(JaxToolboxBaseJobSpecGenStrategy, JobSpecGenStrategy):
    """Job spec generation strategy for JaxToolbox on Kubernetes systems."""

    def gen_job_spec(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        job_name: str,
        num_nodes: int,
        nodes: List[str],
    ) -> Any:
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)

        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_cmd_args["output_path"] = str(output_path)

        combine_threshold_bytes = int(final_env_vars["COMBINE_THRESHOLD"])
        num_nodes = len(nodes) if nodes else num_nodes
        self._handle_threshold_and_env(final_cmd_args, final_env_vars, combine_threshold_bytes, num_nodes)

        xla_flags = self._format_xla_flags(final_cmd_args, "perf")
        final_env_vars["XLA_FLAGS"] = f'"{xla_flags}"'

        # TODO: create run script

        job_spec = self._create_job_spec(job_name, num_nodes, nodes, final_env_vars, final_cmd_args, extra_cmd_args)
        return job_spec

    def _create_job_spec(
        self,
        job_name: str,
        final_num_nodes: int,
        nodes: List[str],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> Dict[Any, Any]:
        """Create the Kubernetes Job specification for JaxToolbox."""
        kubernetes_system = cast(KubernetesSystem, self.system)

        job_spec = {
            "apiVersion": "kubeflow.org/v2beta1",
            "kind": "MPIJob",
            "metadata": {"name": job_name, "namespace": kubernetes_system.default_namespace},
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
                                        "name": "jax-toolbox-launcher",
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
                                "containers": [
                                    {
                                        "image": cmd_args["docker_image_url"],
                                        "name": "jax-toolbox-worker",
                                        "env": self._generate_env_list(env_vars),
                                        "command": ["/bin/bash"],
                                        "args": ["-c", "/usr/sbin/sshd -p 2222; sleep infinity"],
                                        "resources": self._prepare_worker_resources(),
                                        "volumeMounts": [
                                            {"mountPath": "/dev/shm", "name": "dshm"},
                                        ],
                                    }
                                ],
                                "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
                            },
                        },
                    },
                },
            },
        }

        return job_spec

    def _generate_launcher_command(
        self,
        final_num_nodes: int,
        nodes: List[str],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """
        Generate the command that the launcher will execute in Kubernetes.

        This includes running a pretest command, checking for the keyword in the output,
        and conditionally running the main command based on the keyword search result.

        Args:
            final_num_nodes (int): Final number of worker nodes.
            nodes (List[str]): List of specific node names.
            env_vars (Dict[str, str]): Environment variables for the job.
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            extra_cmd_args (str): Additional command-line arguments.

        Returns:
            str: The complete launcher command script as a string.
        """
        pretest_command = self._generate_pre_test_command(cmd_args)
        main_command = self._generate_main_command(env_vars, cmd_args, extra_cmd_args)
        output_file = f"{cmd_args['output_path']}/output_pretest-*.txt"
        keyword = cmd_args.get("keyword", "Avg bus bandwidth")

        # TODO: generate commands conditionally as we did for slurm systems

        return f"""
            # Run pretest command
            {pretest_command}

            # Check the output of pretest for keyword
            file_pattern="{output_file}"
            keyword="{keyword}"
            keyword_found=false

            if grep -q "$keyword" $file_pattern; then
                keyword_found=true
            fi

            # Conditionally run main command
            if [ "$keyword_found" = true ]; then
                {main_command}
            else
                echo "Keyword not found. Skipping main command."
            fi
        """

    def _generate_pre_test_command(self, cmd_args: Dict[str, str]) -> str:
        """
        Generate the pretest command, usually for running NCCL tests.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.

        Returns:
            str: The formatted pretest command.
        """
        pretest_command_parts = [
            f"mpirun --allow-run-as-root -np {cmd_args.get('num_nodes', 2)}",
            "--bind-to none --hostfile /etc/mpi/hostfile",
            self._generate_pre_test_base_command(cmd_args),
        ]
        # TODO: store output and error files and so that CloudAI can take them later

        return " \\\n".join(pretest_command_parts)

    def _generate_main_command(self, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str) -> str:
        """
        Generate the main command to be executed, typically `run.sh`.

        Args:
            env_vars (Dict[str, str]): Final environment variables for the job.
            cmd_args (Dict[str, str]): Final command-line arguments for the job.
            extra_cmd_args (str): Additional command-line arguments.

        Returns:
            str: The formatted main command.
        """
        command_parts = ["/opt/paxml/workspace/run.sh"]

        return (
            f"mpirun --allow-run-as-root -np {cmd_args.get('num_nodes', 2)} "
            "--bind-to none --hostfile /etc/mpi/hostfile "
            f"{' '.join([f'-x {key}={value}' for key, value in env_vars.items()])} "
            f"-o {cmd_args['output_path']}/output-%j-%n-%t.txt "
            f"-e {cmd_args['output_path']}/error-%j-%n-%t.txt "
            " ".join(command_parts)
        )

    def _prepare_launcher_resources(self) -> Dict[str, Dict[str, str]]:
        """
        Prepare resource requests and limits for the launcher container.

        Returns:
            Dict[str, Dict[str, str]]: Resource requests and limits.
        """
        return {"requests": {"cpu": "2", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "8Gi"}}

    def _prepare_worker_resources(self) -> Dict[str, Dict[str, str]]:
        """
        Prepare resource requests and limits for the worker containers.

        Returns:
            Dict[str, Dict[str, str]]: Resource requests and limits.
        """
        return {
            "requests": {"cpu": "24", "memory": "32Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "24", "memory": "32Gi", "nvidia.com/gpu": "1"},
        }

    def _generate_env_list(self, env_vars: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Convert environment variables to a list format compatible with Kubernetes.

        This method formats the environment variables from a dictionary into the structure
        expected by Kubernetes Job specifications.

        Args:
            env_vars (Dict[str, str]): Dictionary of environment variables.

        Returns:
            List[Dict[str, str]]: List of environment variable dictionaries formatted for Kubernetes.
        """
        env_list = [{"name": key, "value": str(value)} for key, value in env_vars.items()]
        return env_list
