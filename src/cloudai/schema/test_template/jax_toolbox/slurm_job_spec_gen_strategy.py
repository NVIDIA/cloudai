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
from typing import Any, Dict, List

from cloudai.systems import SlurmSystem
from .base_job_spec_gen_strategy import JaxToolboxBaseJobSpecGenStrategy


class JaxToolboxSlurmJobSpecGenStrategy(JaxToolboxBaseJobSpecGenStrategy, SlurmJobSpecGenStrategy):
    """Job spec generation strategy for JaxToolbox on Slurm systems."""

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

        slurm_args = self._parse_slurm_args("JaxToolbox", final_env_vars, final_cmd_args, num_nodes, nodes)
        env_vars_str = self._format_env_vars(final_env_vars)
        srun_command = self.generate_full_srun_command(slurm_args, final_env_vars, final_cmd_args, extra_cmd_args)
        return self._write_sbatch_script(slurm_args, env_vars_str, srun_command, output_path)

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        """
        Parse Slurm arguments.

        This method generates a dictionary of Slurm arguments required for the job, including paths, node
        configurations, and container mounts.

        Args:
            job_name_prefix (str): Prefix for the job name.
            env_vars (Dict[str, str]): Environment variables for the job.
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            num_nodes (int): Number of nodes to use.
            nodes (List[str]): List of nodes.

        Returns:
            Dict[str, Any]: Dictionary of Slurm arguments.
        """
        key_prefix = f"{self.test_name}" if self.test_name in ["GPT", "Grok", "Nemotron"] else "common"

        if not all(k in cmd_args for k in [f"{key_prefix}.setup_flags.docker_workspace_dir"]):
            raise ValueError("Required cmd_args keys are missing: docker_workspace_dir")

        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)
        image_path = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            JaxToolboxSlurmInstallStrategy.SUBDIR_PATH,
            JaxToolboxSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        local_workspace_dir = Path(cmd_args["output_path"]).resolve()
        docker_workspace_dir = cmd_args[f"{key_prefix}.setup_flags.docker_workspace_dir"]
        container_mounts = f"{local_workspace_dir}:{docker_workspace_dir}"

        if "pgo_nsys_converter.profile_path" in cmd_args:
            profile_path = Path(cmd_args["pgo_nsys_converter.profile_path"]).resolve()
            container_mounts += f",{profile_path}:{profile_path}"

        base_args.update({"image_path": image_path, "container_mounts": container_mounts})

        output_path = Path(cmd_args["output_path"]).resolve()
        output_suffix = "-%j.txt" if env_vars.get("UNIFIED_STDOUT_STDERR") == "1" else "-%j-%n-%t.txt"
        base_args["output"] = str(output_path / f"output{output_suffix}")
        base_args["error"] = str(output_path / f"error{output_suffix}")

        return base_args

    def generate_full_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> str:
        """ Generate the full srun command for running a job on Slurm. """
        self._create_run_script(slurm_args, env_vars, cmd_args, extra_cmd_args)

        start_container_run = str(cmd_args.get("load_container", "False")).lower() in ("true", "1", "yes")
        output_path = Path(cmd_args["output_path"]).resolve() / "output_pretest-%j-%n-%t.txt"
        error_path = Path(cmd_args["output_path"]).resolve() / "error_pretest-%j-%n-%t.txt"

        commands = []

        pre_test_value = cmd_args.get("pre_test", "False")

        if isinstance(pre_test_value, bool):
            run_pre_test = pre_test_value
        else:
            run_pre_test = str(pre_test_value).lower() in ("true", "1", "yes")

        if run_pre_test:
            pre_test_command = self._generate_pre_test_command(cmd_args, output_path, error_path)
            commands.append(pre_test_command)
            pre_test_check_command = self._generate_pre_test_check_command(cmd_args, output_path)
            commands.append(pre_test_check_command)

        if start_container_run:
            # Load container command
            srun_command_load = self._generate_container_load_srun_command(
                slurm_args, env_vars, cmd_args, extra_cmd_args
            )
            commands.append('if [ "$keyword_found" = true ]; then')
            commands.append('    echo "Loading container with srun command"')
            commands.append(f"    {srun_command_load}")
            commands.append("fi")

        main_srun_command = "\n".join(
            [
                'if [ "$keyword_found" = true ]; then',
                '    echo "Running srun command"',
                "    srun \\",
                "    --mpi=none \\",
                f'    {self.slurm_system.extra_srun_args if self.slurm_system.extra_srun_args else ""} \\',
                "    --export=ALL \\",
                f'    -o {slurm_args["output"]} \\',
                f'    -e {slurm_args["error"]} \\',
                "    --container-name=cont \\",
                f'    --container-mounts={slurm_args["container_mounts"]} \\',
                "    /opt/paxml/workspace/run.sh",
                "fi",
            ]
        )

        # Add the final srun command to the list of commands
        commands.append(main_srun_command)

        # Combine all parts into the final batch script
        full_command = "\n\n".join(commands)

        return full_command

    def _generate_container_load_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> str:
        """
        Generate the srun command to load a container and log the status using Docker commands.

        Args:
            slurm_args (Dict[str, Any]): Dictionary containing the Slurm job settings such as image path.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included in the command.

        Returns:
            str: The generated srun command with proper indentation and logging.
        """
        container_name = "cont"
        container_image = slurm_args["image_path"]

        # Construct the srun command to load the container and check if it's running
        srun_command = "\n".join(
            [
                "",
                "    srun \\",
                "    --mpi=none \\",
                f"    --container-image={container_image} \\",
                f"    --container-name={container_name} \\",
                "    true",
            ]
        )

        return srun_command

    def _generate_pre_test_command(self, cmd_args: Dict[str, Any], output_path: Path, error_path: Path) -> str:
        """
        Generate the pre-test command for running a test.

        Args:
            cmd_args (Dict[str, Any]): A dictionary containing command arguments.
            output_path (Path): The path to the output file.
            error_path (Path): The path to the error file.

        Returns:
            str: The generated pre-test command.
        """
        nccl_test = {k.split(".")[-1]: v for k, v in cmd_args.items() if k.startswith("pre_test.nccl_test")}
        pre_test_command_parts = [
            "srun",
            "--mpi=pmix",
            f"-N {nccl_test.get('num_nodes', 2)}",
            f"-o {output_path}",
            f"-e {error_path}",
            f"--container-image={nccl_test.get('docker_image_url', 'nvcr.io/nvidia/pytorch:24.02-py3')}",
            self._generate_pre_test_base_command(cmd_args)
        ]
        return " \\\n".join(pre_test_command_parts)

    def _generate_pre_test_check_command(self, cmd_args: Dict[str, str], output_path: Path) -> str:
        """
        Generate the command for pre-test check.

        This method generates the command that checks the output of the pre-test
        to determine if the main test should be run.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            output_path (str): The path to the output file.

        Returns:
            str: The generated command for pre-test check.
        """
        directory_path = Path(output_path).parent
        # Create the file pattern with wildcard
        file_pattern = str(directory_path / "output_pretest-*.txt")
        keyword = cmd_args.get("keyword", "Avg bus bandwidth")

        script_lines = [
            f'file_pattern="{file_pattern}"',
            f'keyword="{keyword}"',
            "",
            "# Use grep to search for the keyword in the files",
            'if grep -q "$keyword" $file_pattern; then',
            "    keyword_found=true",
            "fi",
        ]

        script = "\n".join(script_lines)
        return script

    def _create_run_script(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> Path:
        """
        Generate and write the run.sh script to the specified output directory.

        The script configures environment variables, applies necessary command options, and executes the Python command
        within the Slurm environment.

        Args:
            slurm_args (Dict[str, Any]): Slurm arguments including the output path and other job-related settings.
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

        run_script_content = []
        enable_pgle = cmd_args.get(f"{test_name}.enable_pgle", "True")

        do_pgle = enable_pgle if isinstance(enable_pgle, bool) else str(enable_pgle).lower() in ("true", "1", "yes")

        if do_pgle:
            # Prepare environment and script content for the 'profile' stage
            set_xla_flags(test_name, False)
            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "profile")}"'
            profile_content = self._script_content("profile", slurm_args, env_vars, cmd_args, extra_cmd_args)
            run_script_content += profile_content

            set_xla_flags(test_name, True)
            cmd_args[f"{self.test_name}.perf.XLA_FLAGS.xla_gpu_pgle_profile_file_or_directory_path"] = (
                "/opt/paxml/workspace/pgle_output_profile.pbtxt"
            )
            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "perf")}"'
            perf_content = self._script_content("perf", slurm_args, env_vars, cmd_args, extra_cmd_args)
            run_script_content += perf_content
        else:
            set_xla_flags(test_name, True)
            cmd_args[f"{self.test_name}.perf.XLA_FLAGS.xla_gpu_pgle_profile_file_or_directory_path"] = '""'
            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "perf")}"'
            perf_content = self._script_content("perf", slurm_args, env_vars, cmd_args, extra_cmd_args)
            run_script_content += perf_content

        # Write the combined script content to the run.sh file
        run_script_path = Path(cmd_args["output_path"]) / "run.sh"
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
    ) -> List:
        """
        Generate the content of the run script for a given stage.

        This method creates the script lines for a specific stage (e.g., 'profile' or 'perf').

        Args:
            stage (str): The stage of the process ('profile' or 'perf').
            slurm_args (Dict[str, Any]): Slurm job settings.
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
        ]

        script_lines.append(self._generate_python_command(stage, slurm_args, env_vars, cmd_args, extra_cmd_args))
        if self.test_name == "Grok" or self.test_name == "GPT" or self.test_name == "Nemotron":
            script_lines.extend(
                [
                    self._create_pgo_nsys_converter_command(stage),
                ]
            )

        return script_lines

    def _generate_python_command(
        self,
        stage: str,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """
        Construct the complete Python command for execution in the Slurm environment.

        The command is structured with specific ordering of arguments
        to match the operational requirements of the JaxToolbox on Slurm systems.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').
            slurm_args (Dict[str, Any]): Dictionary containing the Slurm job settings such as number of nodes.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included in the Python command.

        Returns:
            str: The formatted Python command string to be executed within a Slurm job.
        """
        fdl_config = cmd_args.get(f"{self.test_name}.fdl_config")
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
        python_command = " \\\n    ".join(parts)

        if stage == "profile":
            python_command += " >> /opt/paxml/workspace/profile_stderr_${SLURM_PROCID}.txt 2>&1"

        nsys_command = (
            "nsys profile \\\n"
            "    -s none \\\n"
            f"    -o /opt/paxml/workspace/nsys_profile_{stage} \\\n"
            "    --force-overwrite true \\\n"
            "    --capture-range=cudaProfilerApi \\\n"
            "    --capture-range-end=stop \\\n"
            "    --cuda-graph-trace=node \\\n"
        )

        slurm_check = (
            'if [ "$SLURM_NODEID" -eq 0 ] && [ "$SLURM_PROCID" -eq 0 ]; then\n'
            f"    {nsys_command}    {python_command}\n"
            "else\n"
            f"    {python_command}\n"
            "fi"
        )

        return slurm_check

    def _create_pgo_nsys_converter_command(self, stage: str) -> str:
        """
        Construct the command to generate the pbtxt file in a multi-line format.

        For readability, extracting required paths from command-line arguments.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').

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
