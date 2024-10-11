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

from cloudai import TestRun
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import JaxToolboxSlurmInstallStrategy


class JaxToolboxSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for JaxToolbox tests on Slurm systems."""

    def __init__(self, system: SlurmSystem, cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, cmd_args)
        self.test_name = ""

    def gen_exec_command(self, tr: TestRun) -> str:
        self.test_name = self._extract_test_name(tr.test.cmd_args)
        self._update_env_vars(tr)

        final_env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.test_definition.extra_env_vars)
        cmd_args = tr.test.test_definition.cmd_args_dict
        cmd_args["output_path"] = str(tr.output_path)
        slurm_args = self._parse_slurm_args("JaxToolbox", final_env_vars, cmd_args, tr.num_nodes, tr.nodes)
        srun_command = self.generate_srun_command(slurm_args, final_env_vars, cmd_args, tr.test.extra_cmd_args)
        return self._write_sbatch_script(slurm_args, final_env_vars, srun_command, tr.output_path)

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

    def _update_env_vars(self, tr: TestRun):
        """Update environment variables."""
        env_vars = tr.test.test_definition.extra_env_vars
        cmd_args = tr.test.test_definition.cmd_args_dict
        num_nodes = len(tr.nodes) if tr.nodes else tr.num_nodes

        self._update_per_gpu_combine_threshold(env_vars, cmd_args, num_nodes)
        self._update_xla_flags(env_vars, cmd_args)

        tr.test.test_definition.extra_env_vars.update(env_vars)

    def _update_per_gpu_combine_threshold(self, env_vars: Dict[str, str], cmd_args: Dict[str, Any], num_nodes: int):
        combine_threshold_bytes = int(env_vars["COMBINE_THRESHOLD"])
        per_gpu_combine_threshold = int(
            combine_threshold_bytes / (int(cmd_args[f"{self.test_name}.setup_flags"]["gpus_per_node"]) * num_nodes)
        )
        env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)

    def _update_xla_flags(self, env_vars: Dict[str, str], cmd_args: Dict[str, Any]):
        env_vars["XLA_FLAGS"] = self._format_xla_flags(cmd_args, "perf")

    def _format_xla_flags(self, cmd_args: Dict[str, Any], stage: str) -> str:
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
        args: dict[str, str] = cmd_args.get(f"{self.test_name}.{stage}", {}).get("XLA_FLAGS", {})

        for flag_name, value in args.items():
            if not flag_name.startswith("xla_"):
                continue
            if isinstance(value, bool):
                value = str(value).lower()
            flag = f"--{flag_name.lower()}={value}"
            xla_flags.append(flag)

        return " ".join(sorted(xla_flags))

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Any],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        key_prefix = f"{self.test_name}" if self.test_name in ["GPT", "Grok", "Nemotron"] else "common"

        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)
        image_path = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            JaxToolboxSlurmInstallStrategy.SUBDIR_PATH,
            JaxToolboxSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        local_workspace_dir = Path(cmd_args["output_path"]).resolve()
        docker_workspace_dir = cmd_args[f"{key_prefix}.setup_flags"]["docker_workspace_dir"]
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

    def generate_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, Any], extra_cmd_args: str
    ) -> str:
        self._create_run_script(env_vars, cmd_args, extra_cmd_args)

        commands = []

        run_pre_test = cmd_args.get("pre_test", {}).get("enable", False)
        if run_pre_test:
            output_path = Path(cmd_args["output_path"]).resolve() / "output_pretest-%j-%n-%t.txt"
            error_path = Path(cmd_args["output_path"]).resolve() / "error_pretest-%j-%n-%t.txt"
            commands.append(self._generate_pre_test_command(cmd_args, output_path, error_path))
            commands.append(self._generate_pre_test_check_command(cmd_args, output_path))
            commands.append('if [ "$PRE_TEST_SUCCESS" = true ]; then')

        load_container = cmd_args.get("load_container", False)
        if load_container:
            commands += self._generate_container_load_command(slurm_args)

        commands += self._generate_run_command(slurm_args)

        if run_pre_test:
            commands.append("fi")

        return "\n".join(commands)

    def _create_run_script(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Any],
        extra_cmd_args: str,
    ) -> Path:
        """
        Generate and write the run.sh script to the specified output directory.

        Args:
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included
                                  in the srun command.

        Returns:
            str: The path to the run.sh script that was created.
        """
        run_script_content = []
        do_pgle = cmd_args.get(f"{self.test_name}.enable_pgle", True)

        if do_pgle:
            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "profile")}"'
            run_script_content += self._script_content("profile", env_vars, cmd_args, extra_cmd_args)

            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "perf")}"'
            run_script_content += self._script_content("perf", env_vars, cmd_args, extra_cmd_args)
        else:
            cmd_args[f"{self.test_name}.perf"]["XLA_FLAGS"]["xla_gpu_pgle_profile_file_or_directory_path"] = '""'
            env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args, "perf")}"'
            run_script_content += self._script_content("perf", env_vars, cmd_args, extra_cmd_args)

        run_script_path = Path(cmd_args["output_path"]) / "run.sh"
        with open(run_script_path, "w") as run_file:
            run_file.write("\n".join(run_script_content))
        run_script_path.chmod(0o755)
        return run_script_path

    def _script_content(
        self,
        stage: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> list:
        """
        Generate the content of the run script for a given stage.

        This method creates the script lines for a specific stage (e.g., 'profile' or 'perf').

        Args:
            stage (str): The stage of the process ('profile' or 'perf').
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

        script_lines.append(self._generate_python_command(stage, cmd_args, extra_cmd_args))
        if self.test_name == "Grok" or self.test_name == "GPT" or self.test_name == "Nemotron":
            script_lines.extend(
                [
                    self._create_pgo_nsys_converter_command(stage, cmd_args),
                ]
            )

        return script_lines

    def _generate_python_command(
        self,
        stage: str,
        cmd_args: Dict[str, Any],
        extra_cmd_args: str,
    ) -> str:
        """
        Construct the PAXML Python command for execution in the Slurm environment.

        Args:
            stage (str): The stage of processing (e.g., 'profile', 'perf').
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
            f"--job_log_dir={cmd_args[f'{self.test_name}.setup_flags']['docker_workspace_dir']}",
            f"--tfds_data_dir={cmd_args[f'{self.test_name}.setup_flags']['tfds_data_dir']}",
            f"--enable_checkpoint_saving={cmd_args[f'{self.test_name}.setup_flags']['enable_checkpoint_saving']}",
            "--multiprocess_gpu",
            "--alsologtostderr",
            f'--fdl_config="{fdl_config}"',
        ]

        fdl_args: Dict[str, str] = cmd_args[f"{self.test_name}.fdl"]

        for key, value in sorted(fdl_args.items()):
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

    def _generate_pre_test_command(self, cmd_args: Dict[str, Any], output_path: Path, error_path: Path) -> str:
        """
        Generate the pre-test command for running a test.

        This method constructs the pre-test command based on the command-line
        arguments provided.

        Args:
            cmd_args (Dict[str, Any]): A dictionary containing command arguments.
            output_path (Path): The path to the output file.
            error_path (Path): The path to the error file.

        Returns:
            str: The generated pre-test command.
        """
        nccl_test = cmd_args.get("pre_test", {}).get("nccl_test", {})
        pre_test_command_parts = [
            "srun",
            "--mpi=pmix",
            f"-N {nccl_test.get('num_nodes', 2)}",
            f"-o {output_path}",
            f"-e {error_path}",
            f"--container-image={nccl_test.get('docker_image_url', 'nvcr.io/nvidia/pytorch:24.02-py3')}",
            f"/usr/local/bin/{nccl_test.get('subtest_name', 'all_gather_perf_mpi')}",
            f"--nthreads {nccl_test.get('nthreads', 1)}",
            f"--ngpus {nccl_test.get('ngpus', 1)}",
            f"--minbytes {nccl_test.get('minbytes', '32M')}",
            f"--maxbytes {nccl_test.get('maxbytes', '16G')}",
            f"--stepbytes {nccl_test.get('stepbytes', '1M')}",
            f"--op {nccl_test.get('op', 'sum')}",
            f"--datatype {nccl_test.get('datatype', 'float')}",
            f"--root {nccl_test.get('root', 0)}",
            f"--iters {nccl_test.get('iters', 20)}",
            f"--warmup_iters {nccl_test.get('warmup_iters', 5)}",
            f"--agg_iters {nccl_test.get('agg_iters', 1)}",
            f"--average {nccl_test.get('average', 1)}",
            f"--parallel_init {nccl_test.get('parallel_init', 0)}",
            f"--check {nccl_test.get('check', 1)}",
            f"--blocking {nccl_test.get('blocking', 0)}",
            f"--cudagraph {nccl_test.get('cudagraph', 0)}",
            f"--stepfactor {nccl_test.get('stepfactor', 2)}",
        ]
        return " \\\n".join(pre_test_command_parts)

    def _generate_pre_test_check_command(self, cmd_args: Dict[str, str], output_path: Path) -> str:
        """
        Generate the command for pre-test check.

        This method generates the command that checks the output of the pre-test to determine if the main test should
        be run.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the job.
            output_path (str): The path to the output file.

        Returns:
            str: The generated command for pre-test check.
        """
        pretest_output_files = str(Path(output_path).parent / "output_pretest-*.txt")
        keyword = cmd_args.get("keyword", "Avg bus bandwidth")

        return "\n".join(
            [
                f'PRETEST_OUTPUT_FILES="{pretest_output_files}"',
                f'keyword="{keyword}"',
                "",
                "# Use grep to search for the keyword in the files",
                'if grep -q "$keyword" $PRETEST_OUTPUT_FILES; then',
                "    PRE_TEST_SUCCESS=true",
                "fi",
            ]
        )

    def _generate_container_load_command(self, slurm_args: Dict[str, Any]) -> List[str]:
        """Generate the command for loading a container."""
        container_image = slurm_args.get("image_path")
        if not container_image:
            raise ValueError("image_path in slurm_args must be a valid path")

        return [
            '    echo "Loading container with srun command"',
            f"    srun --mpi=none --container-image={container_image} --container-name=cont true",
        ]

    def _generate_run_command(self, slurm_args: Dict[str, Any]) -> List[str]:
        """Generate the srun command for executing the test."""
        return [
            '    echo "Running srun command"',
            "    srun \\",
            "    --mpi=none \\",
            f'    {self.system.extra_srun_args if self.system.extra_srun_args else ""} \\',
            "    --export=ALL \\",
            f'    -o {slurm_args["output"]} \\',
            f'    -e {slurm_args["error"]} \\',
            "    --container-name=cont \\",
            f'    --container-mounts={slurm_args["container_mounts"]} \\',
            "    /opt/paxml/workspace/run.sh",
        ]
