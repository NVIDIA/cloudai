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

from typing import Any, Dict


class JaxToolboxBaseJobSpecGenStrategy:
    """Base class for JaxToolbox job specification generation strategies."""

    def __init__(self, system: Any, env_vars: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        """
        Initialize the base job specification generation strategy.

        Args:
            system (Any): The system object (e.g., Slurm, Kubernetes).
            env_vars (Dict[str, Any]): The default environment variables.
            cmd_args (Dict[str, Any]): The default command-line arguments.
        """
        self.test_name = self._extract_test_name(cmd_args)

    def _extract_test_name(self, cmd_args: Dict[str, Any]) -> str:
        """
        Extract the test name from the command-line arguments.

        Args:
            cmd_args (Dict[str, Any]): Command-line arguments for the job.

        Returns:
            str: The extracted test name, capitalized if it's recognized.
        """
        for key in cmd_args:
            if "." in key:
                name = key.split(".")[0]
                if name.lower() in ["grok", "gpt", "nemotron"]:
                    return name.upper() if name.lower() == "gpt" else name.capitalize()
        return ""

    def _handle_threshold_and_env(
        self, cmd_args: Dict[str, str], env_vars: Dict[str, str], combine_threshold_bytes: int, num_nodes: int
    ) -> None:
        """
        Handle threshold and environment variable adjustments based on the test name.

        Adjust environment variables related to XLA thresholds based on the test name
        (e.g., GPT, Grok, Nemotron) and the number of nodes used.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            env_vars (Dict[str, str]): Environment variables for the job.
            combine_threshold_bytes (int): The combine threshold in bytes.
            num_nodes (int): The number of nodes to use.
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

    def _format_xla_flags(self, cmd_args: Dict[str, str], stage: str) -> str:
        """
        Format the XLA_FLAGS environment variable.

        Collect and format XLA-related flags from the command-line arguments for inclusion
        in the job's environment variables.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            stage (str): The stage of the test (e.g., "profile" or "perf").

        Returns:
            str: The formatted XLA_FLAGS string.
        """
        xla_flags = [
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
        ]
        common_prefix = "common.XLA_FLAGS."
        test_prefix = f"{self.test_name}.{stage}.XLA_FLAGS."

        for key, value in cmd_args.items():
            if key.startswith(common_prefix) or key.startswith(test_prefix):
                flag_name = key.split(".")[-1]
                if flag_name.lower() == "xla_gpu_simplify_all_fp_conversions" and value:
                    xla_flags.append(f"--{flag_name.lower()}")
                else:
                    flag = f"--{flag_name.lower()}={'true' if value is True else 'false' if value is False else value}"
                    xla_flags.append(flag)

        return " ".join(xla_flags)

    def _generate_pre_test_base_command(self, cmd_args: Dict[str, str]) -> str:
        """
        Generate the common part of the pretest command, starting from the binary path.

        This method generates the core command for running a pretest, including command-line arguments for parameters like
        nthreads, ngpus, minbytes, and others.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.

        Returns:
            str: The formatted pretest command string, excluding the system-specific command prefix.
        """
        pretest_command_parts = [
            f"/usr/local/bin/{cmd_args.get('pretest_command', 'all_gather_perf_mpi')}",
            f"--nthreads {cmd_args.get('nthreads', 1)}",
            f"--ngpus {cmd_args.get('ngpus', 1)}",
            f"--minbytes {cmd_args.get('minbytes', '8M')}",
            f"--maxbytes {cmd_args.get('maxbytes', '16G')}",
            f"--stepbytes {cmd_args.get('stepbytes', '1M')}",
            f"--op {cmd_args.get('op', 'sum')}",
            f"--datatype {cmd_args.get('datatype', 'float')}",
            f"--root {cmd_args.get('root', 0)}",
            f"--iters {cmd_args.get('iters', 20)}",
            f"--warmup_iters {cmd_args.get('warmup_iters', 5)}",
            f"--agg_iters {cmd_args.get('agg_iters', 1)}",
            f"--average {cmd_args.get('average', 1)}",
            f"--parallel_init {cmd_args.get('parallel_init', 0)}",
            f"--check {cmd_args.get('check', 1)}",
            f"--blocking {cmd_args.get('blocking', 1)}",
            f"--cudagraph {cmd_args.get('cudagraph', 0)}",
            f"--stepfactor {cmd_args.get('stepfactor', 2)}",
        ]
        return " \\\n".join(pretest_command_parts)
