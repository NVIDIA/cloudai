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

import os
from typing import Any, Dict, List

from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import JaxToolboxSlurmInstallStrategy


class JaxToolboxSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for JaxToolbox tests on Slurm systems."""

    def __init__(self, system: SlurmSystem, env_vars: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.test_name = ""

    def gen_exec_command(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        num_nodes: int,
        nodes: List[str],
    ) -> str:
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_cmd_args["output_path"] = output_path

        self.test_name = self._extract_test_name(cmd_args)

        if self.test_name == "GPT":
            # Define the keys to check for the GPT test
            gpt_keys = [
                "GPT.XLA_FLAGS.xla_gpu_all_reduce_combine_threshold_bytes",
                "GPT.XLA_FLAGS.xla_gpu_all_gather_combine_threshold_bytes",
                "GPT.XLA_FLAGS.xla_gpu_reduce_scatter_combine_threshold_bytes",
            ]
            # Find the first key that exists in final_cmd_args
            key = next((k for k in gpt_keys if k in final_cmd_args), None)
            if key is None:
                raise ValueError("None of the GPT specific keys are found in cmd_args.")
        elif self.test_name == "Grok":
            key = f"{self.test_name}.XLA_FLAGS.combine_threshold_bytes"
        else:
            key = "XLA_FLAGS.combine_threshold_bytes"

        combine_threshold_bytes = int(final_env_vars["COMBINE_THRESHOLD"])
        del final_cmd_args[key]

        final_env_vars["COMBINE_THRESHOLD"] = f"{combine_threshold_bytes}"
        num_nodes = len(nodes) if nodes else num_nodes

        setup_flags_key = (
            f"{self.test_name}.setup_flags.gpus_per_node"
            if self.test_name in ["Grok", "GPT"]
            else "common.setup_flags.gpus_per_node"
        )
        per_gpu_combine_threshold = int(combine_threshold_bytes / (int(final_cmd_args[setup_flags_key]) * num_nodes))
        final_env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)

        xla_flags = self._format_xla_flags(final_cmd_args)
        final_env_vars["XLA_FLAGS"] = f'"{xla_flags}"'

        env_vars_str = self._format_env_vars(final_env_vars)

        slurm_args = self._parse_slurm_args("JaxToolbox", final_env_vars, final_cmd_args, num_nodes, nodes)
        srun_command = self.generate_full_srun_command(slurm_args, final_env_vars, final_cmd_args, extra_cmd_args)
        return self._write_sbatch_script(slurm_args, env_vars_str, srun_command, output_path)

    def _extract_test_name(self, cmd_args: Dict[str, Any]) -> str:
        test_name = ""
        for key in cmd_args:
            if "." in key:
                name = key.split(".")[0]
                if name.lower() == "grok":
                    test_name = "Grok"
                elif name.lower() == "gpt":
                    test_name = "GPT"
        return test_name

    def _format_xla_flags(self, cmd_args: Dict[str, str]) -> str:
        """
        Format the XLA_FLAGS environment variable.

        Done by extracting all command-line arguments prefixed with 'common.XLA_FLAGS' or '{test_name}.XLA_FLAGS'
        and concatenating them into a single string with the appropriate formatting for execution.

        Args:
            test_name (str): The name of the test (e.g., "GPT" or "Grok").
            cmd_args (Dict[str, str]): Command-line arguments.

        Returns:
            str: A single string containing all XLA-related flags formatted for inclusion in the environment variables.
        """
        xla_flags = []

        # Standard flags that are always included
        if self.test_name == "Grok":
            xla_flags.extend(
                [
                    "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                    "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                ]
            )
        elif self.test_name == "GPT":
            xla_flags.extend(
                [
                    "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                    "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                    "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                ]
            )

        # Prefixes for common and test-specific XLA flags
        common_prefix = "common.XLA_FLAGS."
        test_prefix = f"{self.test_name}.XLA_FLAGS."

        for key, value in cmd_args.items():
            # Check if the key starts with either common or test-specific prefix
            if key.startswith(common_prefix) or key.startswith(test_prefix):
                # Extract the flag name from the key
                flag_name = key.split(".")[-1]
                # Check if the flag is 'xla_gpu_simplify_all_fp_conversions'
                if flag_name.lower() == "xla_gpu_simplify_all_fp_conversions":
                    # For this specific flag, append only the flag name if the value is True
                    if value:
                        xla_flags.append(f"--{flag_name.lower()}")
                else:
                    # For all other flags, format the flag with its value, appending boolean values as is
                    flag = f"--{flag_name.lower()}={'true' if value is True else 'false' if value is False else value}"
                    xla_flags.append(flag)
        return " ".join(xla_flags)

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        # Determine the key prefix based on test_name
        key_prefix = f"{self.test_name}" if self.test_name in ["GPT", "Grok"] else "common"

        # Adjusted the key to use the dynamic key_prefix
        if not all(k in cmd_args for k in [f"{key_prefix}.setup_flags.docker_workspace_dir"]):
            raise ValueError("Required cmd_args keys are missing: docker_workspace_dir")

        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)
        image_path = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            JaxToolboxSlurmInstallStrategy.SUBDIR_PATH,
            JaxToolboxSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        local_workspace_dir = os.path.abspath(cmd_args["output_path"])
        # Use the dynamic key_prefix for accessing docker_workspace_dir
        docker_workspace_dir = cmd_args[f"{key_prefix}.setup_flags.docker_workspace_dir"]
        container_mounts = f"{local_workspace_dir}:{docker_workspace_dir}"

        if "pgo_nsys_converter.profile_path" in cmd_args:
            profile_path = cmd_args["pgo_nsys_converter.profile_path"]
            container_mounts += f",{profile_path}:{profile_path}"

        base_args.update({"image_path": image_path, "container_mounts": container_mounts})

        output_path = os.path.abspath(cmd_args["output_path"])
        output_suffix = "-%j.txt" if env_vars.get("UNIFIED_STDOUT_STDERR") == "1" else "-%j-%n-%t.txt"
        base_args["output"] = os.path.join(output_path, f"output{output_suffix}")
        base_args["error"] = os.path.join(output_path, f"error{output_suffix}")

        return base_args

    def generate_full_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> str:
        self._create_run_script(slurm_args, env_vars, cmd_args, extra_cmd_args)

        srun_command_parts = [
            "srun",
            f"--mpi={self.slurm_system.mpi}",
            f"{self.slurm_system.extra_srun_args if self.slurm_system.extra_srun_args else ''}",
            "--export=ALL",
            f"-o {slurm_args['output']}",
            f"-e {slurm_args['error']}",
            f"--container-image={slurm_args['image_path']}",
            f"--container-mounts={slurm_args['container_mounts']}",
            "/opt/paxml/workspace/run.sh",
        ]

        return " \\\n".join(srun_command_parts)

    def _create_run_script(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """
        Generate and writes the run.sh script to the specified output directory.

        The script configures environment variables, applies necessary command options, and executes the Python command
        within the SLURM environment.

        Args:
            test_name (str): The name of the test being run.
            slurm_args (Dict[str, Any]): SLURM arguments including the output path and other job-related settings.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included
                                  in the srun command.

        Returns:
            str: The path to the run.sh script that was created.
        """
        test_name = self.test_name

        def set_xla_flags(test_name: str, profile_enabled: bool):
            """Set the XLA_FLAGS for profiling or performance based on the stage."""
            flags = [
                "xla_gpu_enable_latency_hiding_scheduler",
            ]

            state = "true" if profile_enabled else "false"
            for flag in flags:
                cmd_args[f"{test_name}.XLA_FLAGS.{flag}"] = state

        # Prepare environment and script content for the 'profile' stage
        set_xla_flags(test_name, False)
        env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args)}"'

        profile_content = self._script_content("profile", slurm_args, env_vars, cmd_args, extra_cmd_args)

        # Prepare environment and script content for the 'perf' stage
        set_xla_flags(test_name, True)
        cmd_args[f"{self.test_name}.XLA_FLAGS.xla_gpu_pgle_profile_file_or_directory_path"] = (
            "/opt/paxml/workspace/pgle_output_profile.pbtxt"
        )
        env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args)}"'

        perf_content = self._script_content("perf", slurm_args, env_vars, cmd_args, extra_cmd_args)

        # Combine both parts into the run script content
        run_script_content = profile_content + perf_content
        run_script_path = os.path.join(cmd_args["output_path"], "run.sh")
        with open(run_script_path, "w") as run_file:
            run_file.write("\n".join(run_script_content))
        os.chmod(run_script_path, 0o755)
        return run_script_path

    def _script_content(
        self,
        stage: str,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> list:
        """
        Generate the content of the run script for a given stage.

        Args:
            stage (str): The stage of the process ('profile' or 'perf').
            slurm_args (Dict[str, Any]): SLURM job settings.
            env_vars (Dict[str, str]): Environment variables for the job.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments.

        Returns:
            list: Lines of the script for the given stage.
        """
        script_lines = [
            "#!/bin/bash" if stage == "profile" else "",
            "",
            self._format_env_vars(env_vars),
            "",
            self._generate_python_command(stage, slurm_args, env_vars, cmd_args, extra_cmd_args),
        ]

        if self.test_name == "Grok" or self.test_name == "GPT":
            script_lines.extend(
                [
                    self._create_pgo_nsys_converter_command(stage, cmd_args),
                    self._create_nsys_to_sqlite_command(stage, cmd_args),
                ]
            )

        return script_lines

    def _combine_fdl_flags(self, cmd_args: Dict[str, str], test_name: str) -> Dict[str, str]:
        combined_fdl_args = {}
        # Combine common.fdl flags
        common_prefix = "common.fdl."
        for key, value in cmd_args.items():
            if key.startswith(common_prefix):
                flag_name = key[len(common_prefix) :]
                combined_fdl_args[flag_name] = value

        # Override with test_name.fdl flags
        test_prefix = f"{test_name}.fdl."
        for key, value in cmd_args.items():
            if key.startswith(test_prefix):
                flag_name = key[len(test_prefix) :]
                combined_fdl_args[flag_name] = value

        # Now combined_fdl_args contains all fdl flags, with test_name.fdl overriding common.fdl where applicable
        return combined_fdl_args

    def _get_fdl_config_value(self, cmd_args):
        # Format the key based on the test_name
        key = f"{self.test_name}.fdl_config"

        # Access and return the value from cmd_args
        return cmd_args.get(key)

    def _generate_python_command(
        self,
        stage: str,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """
        Construct the complete Python command for execution in the SLURM environment.

        The command is structured with specific ordering of arguments
        to match the operational requirements of the JaxToolbox on Slurm systems.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').
            test_name (str): The name of the test being run.
            slurm_args (Dict[str, Any]): Dictionary containing the SLURM job settings such as number of nodes.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included in the Python command.

        Returns:
            str: The formatted Python command string to be executed within a
                 SLURM job.
        """
        fdl_config = self._get_fdl_config_value(cmd_args)
        parts = [
            "python3 -u -m paxml.main",
            "--num_hosts=$SLURM_NTASKS",
            "--server_addr=$SLURM_JOB_MASTER_NODE:12345",
            "--host_idx=$SLURM_PROCID",
            f"--job_log_dir={cmd_args[f'{self.test_name}.setup_flags.docker_workspace_dir']}",
            f"--tfds_data_dir={cmd_args[f'{self.test_name}.setup_flags.tfds_data_dir']}",
            f"--enable_checkpoint_saving={cmd_args[f'{self.test_name}.setup_flags.enable_checkpoint_saving']}",
            "--multiprocess_gpu",
            "--alsologtostderr",
            f'--fdl_config="{fdl_config}"',
        ]

        # Dynamically adding fdl. prefixed arguments
        fdl_prefix = f"{self.test_name}.fdl."
        fdl_args = {k[len(fdl_prefix) :]: v for k, v in cmd_args.items() if k.startswith(fdl_prefix)}

        for key, value in fdl_args.items():
            parts.append(f"--fdl.{key.upper()}={value}")
        if extra_cmd_args:
            parts.append(extra_cmd_args)
        python_command = " \\\n".join(parts)

        if stage == "profile":
            python_command += " >> /opt/paxml/workspace/profile_stderr.txt 2>&1"

        nsys_command = (
            "nsys profile \\\n"
            "-s none \\\n"
            f"-o /opt/paxml/workspace/nsys_profile_{stage} \\\n"
            "--force-overwrite true \\\n"
            "--capture-range=cudaProfilerApi \\\n"
            "--capture-range-end=stop \\\n"
            "--cuda-graph-trace=node \\\n"
        )
        python_command = nsys_command + python_command

        return python_command

    def _create_pgo_nsys_converter_command(self, stage: str, cmd_args: Dict[str, str]) -> str:
        """
        Construct the command to generate the pbtxt file in a multi-line format.

        For readability, extracting required paths from command-line arguments.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').
            cmd_args (Dict[str, str]): Command-line arguments containing paths and configurations.

        Returns:
            List[str]: The command split into multiple lines for clarity, enclosed in a conditional check.
        """
        base_command = "python /opt/jax/jax/tools/pgo_nsys_converter.py"
        args = [
            f"--profile_path /opt/paxml/workspace/nsys_profile_{stage}.nsys-rep",
            "--post_process",
            f"--pgle_output_path /opt/paxml/workspace/pgle_output_{stage}.pbtxt",
        ]
        command = " \\\n    ".join([base_command] + args)
        command += " > /dev/null 2>&1"

        return "\n".join(
            ["", 'if [ "$SLURM_NODEID" -eq 0 ] && [ "$SLURM_PROCID" -eq 0 ]; then', f"    {command}", "fi"]
        )

    def _create_nsys_to_sqlite_command(self, stage: str, cmd_args: Dict[str, str]) -> str:
        """
        Construct the command to convert the nsys profile file to an sqlite file.

        This command is to be executed conditionally on the master node only.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').
            cmd_args (Dict[str, str]): Command-line arguments.

        Returns:
            List[str]: The command split into multiple lines for clarity, enclosed in a conditional check.
        """
        base_command = "nsys export"
        args = [
            f"/opt/paxml/workspace/nsys_profile_{stage}.nsys-rep",
            f"--output /opt/paxml/workspace/nsys_profile_{stage}.sqlite",
            "--type sqlite",
        ]
        command = " \\\n    ".join([base_command] + args)
        command += " > /dev/null 2>&1"

        return "\n".join(
            ["", 'if [ "$SLURM_NODEID" -eq 0 ] && [ "$SLURM_PROCID" -eq 0 ]; then', f"    {command}", "fi"]
        )
