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


class JaxToolboxGptCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for GPT tests on Slurm systems."""

    @staticmethod
    def supports(test_name: str) -> bool:
        """
        Determine if this strategy supports the given test name.

        Args:
            test_name (str): The name of the test (e.g., "gpt").

        Returns:
            bool: True if this strategy supports the test name, False otherwise.
        """
        return test_name == "gpt"

    def __init__(self, system: SlurmSystem, env_vars: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.test_name = "GPT"

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
        # Override environment variables and command-line arguments
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_cmd_args["output_path"] = output_path

        # GPT-specific logic for handling thresholds
        gpt_keys = [
            "GPT.XLA_FLAGS.xla_gpu_all_reduce_combine_threshold_bytes",
            "GPT.XLA_FLAGS.xla_gpu_all_gather_combine_threshold_bytes",
            "GPT.XLA_FLAGS.xla_gpu_reduce_scatter_combine_threshold_bytes",
        ]
        key = next((k for k in gpt_keys if k in final_cmd_args), None)
        if key is None:
            raise ValueError("None of the GPT specific keys are found in cmd_args.")

        combine_threshold_bytes = int(final_env_vars["COMBINE_THRESHOLD"])
        del final_cmd_args[key]
        final_env_vars["COMBINE_THRESHOLD"] = f"{combine_threshold_bytes}"

        num_nodes = len(nodes) if nodes else num_nodes
        setup_flags_key = f"{self.test_name}.setup_flags.gpus_per_node"
        per_gpu_combine_threshold = int(combine_threshold_bytes / (int(final_cmd_args[setup_flags_key]) * num_nodes))
        final_env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)

        xla_flags = self._format_xla_flags(final_cmd_args)
        final_env_vars["XLA_FLAGS"] = f'"{xla_flags}"'

        env_vars_str = self._format_env_vars(final_env_vars)

        slurm_args = self._parse_slurm_args("JaxToolbox", final_env_vars, final_cmd_args, num_nodes, nodes)
        srun_command = self.generate_full_srun_command(slurm_args, final_env_vars, final_cmd_args, extra_cmd_args)
        return self._write_sbatch_script(slurm_args, env_vars_str, srun_command, output_path)

    def _format_xla_flags(self, cmd_args: Dict[str, str]) -> str:
        """Format XLA_FLAGS specific to GPT."""
        xla_flags = [
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
        ]
        common_prefix = "common.XLA_FLAGS."
        test_prefix = f"{self.test_name}.XLA_FLAGS."

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

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        """Parse command arguments to configure Slurm job settings specifically for GPT."""
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

        # Ensure required cmd_args for GPT
        key_prefix = f"{self.test_name}"
        if not all(k in cmd_args for k in [f"{key_prefix}.setup_flags.docker_workspace_dir"]):
            raise ValueError("Required cmd_args keys are missing: docker_workspace_dir")

        image_path = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            JaxToolboxSlurmInstallStrategy.SUBDIR_PATH,
            JaxToolboxSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        local_workspace_dir = os.path.abspath(cmd_args["output_path"])
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
        """Generate the full srun command for running a GPT job on Slurm."""
        self._create_run_script(slurm_args, env_vars, cmd_args, extra_cmd_args)

        output_path = os.path.join(os.path.abspath(cmd_args["output_path"]), "output_pretest-%j-%n-%t.txt")
        error_path = os.path.join(os.path.abspath(cmd_args["output_path"]), "error_pretest-%j-%n-%t.txt")

        commands = []

        run_pre_test = cmd_args.get("pre_test", "False").lower() in ("true", "1", "yes")

        if run_pre_test:
            pre_test_command = self._generate_pre_test_command(cmd_args, output_path, error_path)
            commands.append(pre_test_command)
            pre_test_check_command = self._generate_pre_test_check_command(cmd_args, output_path)
            commands.append(pre_test_check_command)

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

        srun_command = " \\\n".join(srun_command_parts).strip()

        if run_pre_test:
            srun_command = f'if [ "$keyword_found" = true ]; then\n{srun_command}\nfi'

        commands.append(srun_command)

        return "\n\n".join(commands)

    def _generate_pre_test_command(self, cmd_args: Dict[str, Any], output_path: str, error_path: str) -> str:
        """Generate the pre-test command for GPT."""
        nccl_test = {k.split(".")[-1]: v for k, v in cmd_args.items() if k.startswith("pre_test.nccl_test")}
        pre_test_command_parts = [
            "srun",
            "--mpi=pmix",
            f"-o {output_path}",
            f"-e {error_path}",
            f"--container-image={nccl_test.get('docker_image_url', 'nvcr.io/nvidia/pytorch:24.02-py3')}",
            f"/usr/local/bin/{nccl_test.get('preset', 'all_gather_perf_mpi')}",
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

    def _generate_pre_test_check_command(self, cmd_args: Dict[str, str], output_path: str) -> str:
        """Generate the command for pre-test check for GPT."""
        directory_path = os.path.dirname(output_path)
        file_pattern = os.path.join(directory_path, "output_pretest-*.txt")
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

        return "\n".join(script_lines)

    def _create_run_script(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """Generate and write the run.sh script specific to GPT."""
        test_name = self.test_name

        def set_xla_flags(test_name: str, profile_enabled: bool):
            """Set the XLA_FLAGS for profiling or performance based on the stage."""
            flags = [
                "xla_gpu_enable_latency_hiding_scheduler",
            ]

            state = "true" if profile_enabled else "false"
            for flag in flags:
                cmd_args[f"{test_name}.XLA_FLAGS.{flag}"] = state

        set_xla_flags(test_name, False)
        env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args)}"'

        profile_content = self._script_content("profile", slurm_args, env_vars, cmd_args, extra_cmd_args)

        set_xla_flags(test_name, True)
        cmd_args[f"{self.test_name}.XLA_FLAGS.xla_gpu_pgle_profile_file_or_directory_path"] = (
            "/opt/paxml/workspace/pgle_output_profile.pbtxt"
        )
        env_vars["XLA_FLAGS"] = f'"{self._format_xla_flags(cmd_args)}"'

        perf_content = self._script_content("perf", slurm_args, env_vars, cmd_args, extra_cmd_args)

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
        """Generate the content of the run script for a given stage."""
        script_lines = [
            "#!/bin/bash" if stage == "profile" else "",
            "",
            self._format_env_vars(env_vars),
            "",
        ]

        script_lines.append(self._generate_python_command(stage, slurm_args, env_vars, cmd_args, extra_cmd_args))
        script_lines.extend(
            [
                self._create_pgo_nsys_converter_command(stage, cmd_args),
                self._create_nsys_to_sqlite_command(stage, cmd_args),
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
        """Construct the complete Python command for GPT execution in the Slurm environment."""
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
        """Construct the command to generate the pbtxt file for GPT."""
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
        """Construct the command to convert the nsys profile file to sqlite for GPT."""
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

    def _get_fdl_config_value(self, cmd_args):
        """Retrieve the FDL configuration value for GPT."""
        key = f"{self.test_name}.fdl_config"
        return cmd_args.get(key)
