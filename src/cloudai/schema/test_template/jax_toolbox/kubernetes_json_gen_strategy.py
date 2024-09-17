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

from cloudai import JsonGenStrategy
from cloudai.systems import KubernetesSystem


class JaxToolboxKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for JaxToolbox on Kubernetes systems."""

    def __init__(self, system: KubernetesSystem, env_vars: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.test_name = ""

    def gen_json(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        job_name: str,
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[Any, Any]:
        self.test_name = self._extract_test_name(cmd_args)

        # Combine env_vars and extra_env_vars
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)

        # Combine command arguments
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_cmd_args["output_path"] = str(output_path)

        # Handle threshold logic and set environment variables accordingly
        combine_threshold_bytes = int(final_env_vars["COMBINE_THRESHOLD"])
        num_nodes = len(nodes) if nodes else num_nodes
        self._handle_threshold_and_env(final_cmd_args, final_env_vars, combine_threshold_bytes, num_nodes)

        # Format XLA_FLAGS and assign to final_env_vars
        xla_flags = self._format_xla_flags(final_cmd_args, "perf")
        final_env_vars["XLA_FLAGS"] = f'"{xla_flags}"'

        # Create the Kubernetes job spec
        job_spec = self._create_job_spec(job_name, num_nodes, nodes, final_env_vars, final_cmd_args, extra_cmd_args)

        return job_spec

    def _handle_threshold_and_env(
        self, cmd_args: Dict[str, str], env_vars: Dict[str, str], combine_threshold_bytes: int, num_nodes: int
    ):
        """
        Handle threshold and environment variable adjustments based on the test name.

        This method handles the test-specific logic for setting the correct thresholds
        and environment variables based on the type of test being run (e.g., GPT, Grok, Nemotron).

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            env_vars (Dict[str, str]): Environment variables for the job.
            combine_threshold_bytes (int): The combine threshold in bytes.
            num_nodes (int): Number of nodes to use.
        """
        if self.test_name in ["GPT", "Nemotron"]:
            keys = [
                f"{self.test_name}.XLA_FLAGS.xla_gpu_all_reduce_combine_threshold_bytes",
                f"{self.test_name}.XLA_FLAGS.xla_gpu_all_gather_combine_threshold_bytes",
                f"{self.test_name}.XLA_FLAGS.xla_gpu_reduce_scatter_combine_threshold_bytes",
            ]
            key = next((k for k in keys if k in cmd_args), None)
            if key is None:
                raise ValueError(f"None of the {self.test_name} specific keys are found in cmd_args.")

        elif self.test_name == "Grok":
            key = f"{self.test_name}.perf.XLA_FLAGS.combine_threshold_bytes"
        else:
            key = "XLA_FLAGS.combine_threshold_bytes"

        del cmd_args[key]

        # Determine the per-GPU combine threshold based on the number of nodes and GPUs per node
        setup_flags_key = (
            f"{self.test_name}.setup_flags.gpus_per_node"
            if self.test_name in ["Grok", "GPT", "Nemotron"]
            else "common.setup_flags.gpus_per_node"
        )
        per_gpu_combine_threshold = int(combine_threshold_bytes / (int(cmd_args[setup_flags_key]) * num_nodes))
        env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)

    def _extract_test_name(self, cmd_args: Dict[str, Any]) -> str:
        """
        Extract the test name from the command-line arguments.

        This method identifies the test name (e.g., GPT, Grok, Nemotron) by examining
        the command-line arguments.

        Args:
            cmd_args (Dict[str, Any]): Command-line arguments for the job.

        Returns:
            str: The name of the test (capitalized).
        """
        for key in cmd_args:
            if "." in key:
                name = key.split(".")[0]
                if name.lower() in ["grok", "gpt", "nemotron"]:
                    return name.upper() if name.lower() == "gpt" else name.capitalize()
        return ""

    def _format_xla_flags(self, cmd_args: Dict[str, str], stage: str) -> str:
        """
        Format the XLA_FLAGS environment variable.

        This method extracts all command-line arguments prefixed with 'common.XLA_FLAGS'
        or '{test_name}.{stage}.XLA_FLAGS' and concatenates them into a single string formatted
        for execution.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            stage (str): The stage of the test, can be "profile" or "perf".

        Returns:
            str: A single string containing all XLA-related flags formatted for inclusion
                    in the environment variables.
        """
        xla_flags = [
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
        ]
        # Prefixes for common and test-specific XLA flags
        common_prefix = "common.XLA_FLAGS."
        test_prefix = f"{self.test_name}.{stage}.XLA_FLAGS."

        for key, value in cmd_args.items():
            if key.startswith(common_prefix) or key.startswith(test_prefix):
                flag_name = key.split(".")[-1]
                if flag_name.lower() == "xla_gpu_simplify_all_fp_conversions":
                    if value:
                        xla_flags.append(f"--{flag_name.lower()}")
                else:
                    flag = f"--{flag_name.lower()}={'true' if value is True else 'false' if value is False else value}"
                    xla_flags.append(flag)

        return " ".join(xla_flags)

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
                                            self._generate_main_command(env_vars, cmd_args, extra_cmd_args),
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

    def _generate_main_command(self, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str) -> str:
        """Generate the main command to run (e.g., run.sh)."""
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
        """Prepare resource requests and limits for the launcher container."""
        return {"requests": {"cpu": "2", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "8Gi"}}

    def _prepare_worker_resources(self) -> Dict[str, Dict[str, str]]:
        """Prepare resource requests and limits for the worker containers."""
        return {
            "requests": {"cpu": "24", "memory": "32Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "24", "memory": "32Gi", "nvidia.com/gpu": "1"},
        }

    def _generate_env_list(self, env_vars: Dict[str, str]) -> List[Dict[str, str]]:
        """Convert environment variables to a list format compatible with Kubernetes."""
        env_list = [{"name": key, "value": str(value)} for key, value in env_vars.items()]
        return env_list
