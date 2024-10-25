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

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cloudai import CommandGenStrategy, TestRun, TestScenario
from cloudai.systems import SlurmSystem


class SlurmCommandGenStrategy(CommandGenStrategy):
    """
    Abstract base class for defining command generation strategies specific to Slurm environments.

    Attributes
        system (SlurmSystem): A casted version of the `system` attribute, which provides Slurm-specific
            properties and methods.
    """

    def __init__(self, system: SlurmSystem, cmd_args: Dict[str, Any]) -> None:
        """
        Initialize a new SlurmCommandGenStrategy instance.

        Args:
            system (SlurmSystem): The system schema object.
            cmd_args (Dict[str, Any]): Command-line arguments.
        """
        super().__init__(system, cmd_args)
        self.system = system
        if not self.system.default_partition:
            raise ValueError(
                "Default partition not set in the Slurm system object. "
                "The 'default_partition' attribute should be properly defined in the Slurm system configuration. "
                "Please ensure that 'default_partition' is set correctly in the corresponding system configuration "
                "(e.g., system.toml)."
            )

        self.docker_image_url = self.cmd_args.get("docker_image_url", "")

    def gen_exec_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        slurm_args = self._parse_slurm_args(tr.test.test_template.__class__.__name__, env_vars, cmd_args, tr)

        if tr.prologue:
            prologue_command = self.gen_prologue(tr.prologue, tr.output_path)
            srun_command = self._gen_srun_command(slurm_args, env_vars, cmd_args, tr.test.extra_cmd_args)
            command_list = [prologue_command, "if [ $PROLOGUE_SUCCESS -eq 1 ]; then", f"    {srun_command}"]

            if tr.epilogue:
                epilogue_command = self.gen_epilogue(tr.epilogue, tr.output_path)
                command_list.append(f"    {epilogue_command}")

            command_list.append("fi")
        else:
            srun_command = self._gen_srun_command(slurm_args, env_vars, cmd_args, tr.test.extra_cmd_args)
            command_list = [srun_command]

            if tr.epilogue:
                epilogue_command = self.gen_epilogue(tr.epilogue, tr.output_path)
                command_list.append(epilogue_command)

        full_command = "\n".join(command_list).strip()
        return self._write_sbatch_script(slurm_args, env_vars, full_command, tr.output_path)

    def gen_srun_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        slurm_args = self._parse_slurm_args(tr.test.test_template.__class__.__name__, env_vars, cmd_args, tr)
        return self._gen_srun_command(slurm_args, env_vars, cmd_args, tr.test.extra_cmd_args)

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, str], tr: TestRun
    ) -> Dict[str, Any]:
        """
        Parse command arguments to configure Slurm job settings.

        Args:
            job_name_prefix (str): Prefix for the job name.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            tr (TestRun): Test run object.

        Returns:
            Dict[str, Any]: Dictionary containing configuration for Slurm job.

        Raises:
            KeyError: If partition or essential node settings are missing.
        """
        job_name = self.job_name(job_name_prefix)

        parsed_nodes = self.system.parse_nodes(tr.nodes)
        num_nodes = len(parsed_nodes) if parsed_nodes else tr.num_nodes
        node_list_str = ",".join(parsed_nodes) if parsed_nodes else ""

        slurm_args = {
            "job_name": job_name,
            "num_nodes": num_nodes,
            "node_list_str": node_list_str,
        }
        if "time_limit" in cmd_args:
            slurm_args["time_limit"] = cmd_args["time_limit"]

        return slurm_args

    def job_name(self, job_name_prefix: str) -> str:
        job_name = f"{job_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.system.account:
            job_name = f"{self.system.account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return job_name

    def gen_prologue(self, prologue: TestScenario, base_output_path: Path) -> str:
        """
        Generate the prologue command by running all tests defined in the prologue test scenario.

        Args:
            prologue (TestScenario): The prologue test scenario containing the tests to be run.
            base_output_path (Path): The base output directory path for storing prologue outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the prologue.
        """
        if not prologue.test_runs:
            return "PROLOGUE_SUCCESS=1\n"

        prologue_output_dir = base_output_path / "prologue"
        prologue_output_dir.mkdir(parents=True, exist_ok=True)

        prologue_commands = []
        success_vars = []

        for idx, tr in enumerate(prologue.test_runs):
            plugin_dir = prologue_output_dir / tr.test.name
            plugin_dir.mkdir(parents=True, exist_ok=True)
            tr.output_path = plugin_dir

            srun_command = tr.test.test_template.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={plugin_dir / 'stdout.txt'} --error={plugin_dir / 'stderr.txt'} "
            )
            prologue_commands.append(srun_command_with_output)

            success_var = f"SUCCESS_{idx}"
            success_vars.append(success_var)

            success_check_command = tr.test.test_template.gen_srun_success_check(tr)
            prologue_commands.append(f"{success_var}=$({success_check_command})")

        combined_success_var = " && ".join([f"[ ${var} -eq 1 ]" for var in success_vars])

        prologue_commands.append(f"PROLOGUE_SUCCESS=$( {combined_success_var} && echo 1 || echo 0 )")

        return "\n".join(prologue_commands)

    def gen_epilogue(self, epilogue: TestScenario, base_output_path: Path) -> str:
        """
        Generate the epilogue command by running all tests defined in the epilogue test scenario.

        Args:
            epilogue (TestScenario): The epilogue test scenario containing the tests to be run.
            base_output_path (Path): The base output directory path for storing epilogue outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the epilogue.
        """
        if not epilogue.test_runs:
            return ""

        epilogue_output_dir = base_output_path / "epilogue"
        epilogue_output_dir.mkdir(parents=True, exist_ok=True)

        epilogue_commands = []

        for tr in epilogue.test_runs:
            plugin_dir = epilogue_output_dir / tr.test.name
            plugin_dir.mkdir(parents=True, exist_ok=True)
            tr.output_path = plugin_dir

            srun_command = tr.test.test_template.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={plugin_dir / 'stdout.txt'} --error={plugin_dir / 'stderr.txt'} "
            )
            epilogue_commands.append(srun_command_with_output)

        return "\n".join(epilogue_commands)

    def _gen_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> str:
        srun_command_parts = self.gen_srun_prefix(slurm_args)
        test_command_parts = self.generate_test_command(env_vars, cmd_args, extra_cmd_args)
        return " ".join(srun_command_parts + test_command_parts)

    def gen_srun_prefix(self, slurm_args: Dict[str, Any]) -> List[str]:
        srun_command_parts = ["srun", f"--mpi={self.system.mpi}"]
        if slurm_args.get("image_path"):
            srun_command_parts.append(f'--container-image={slurm_args["image_path"]}')
            if slurm_args.get("container_mounts"):
                srun_command_parts.append(f'--container-mounts={slurm_args["container_mounts"]}')

        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        return []

    def _add_reservation(self, batch_script_content: List[str]):
        """
        Add reservation if provided.

        Args:
            batch_script_content (List[str]): content of the batch script.

        Returns:
            List[str]: updated batch script with reservation if exists.
        """
        reservation_key = "--reservation "
        if self.system.extra_srun_args and reservation_key in self.system.extra_srun_args:
            reservation = self.system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            batch_script_content.append(f"#SBATCH --reservation={reservation}")

        return batch_script_content

    def _write_sbatch_script(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], srun_command: str, output_path: Path
    ) -> str:
        """
        Write the batch script for Slurm submission and return the sbatch command.

        Args:
            slurm_args (Dict[str, Any]): Slurm-specific arguments.
            env_vars (env_vars: Dict[str, str]): Environment variables.
            srun_command (str): srun command.
            output_path (Path): Output directory for script and logs.

        Returns:
            str: sbatch command to submit the job.
        """
        batch_script_content = [
            "#!/bin/bash",
            f"#SBATCH --job-name={slurm_args['job_name']}",
            f"#SBATCH -N {slurm_args['num_nodes']}",
        ]

        self._append_sbatch_directives(batch_script_content, slurm_args, output_path)

        env_vars_str = self._format_env_vars(env_vars)
        batch_script_content.extend([env_vars_str, "", srun_command])

        batch_script_path = output_path / "cloudai_sbatch_script.sh"
        with batch_script_path.open("w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"

    def _append_sbatch_directives(
        self, batch_script_content: List[str], args: Dict[str, Any], output_path: Path
    ) -> None:
        """
        Append SBATCH directives to the batch script content.

        Args:
            batch_script_content (List[str]): The list of script lines to append to.
            args (Dict[str, Any]): Arguments including job settings.
            output_path (Path): Output directory for script and logs.
        """
        batch_script_content = self._add_reservation(batch_script_content)

        if "output" not in args:
            batch_script_content.append(f"#SBATCH --output={output_path / 'stdout.txt'}")
        if "error" not in args:
            batch_script_content.append(f"#SBATCH --error={output_path / 'stderr.txt'}")
        batch_script_content.append(f"#SBATCH --partition={self.system.default_partition}")
        if args["node_list_str"]:
            batch_script_content.append(f"#SBATCH --nodelist={args['node_list_str']}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")
        if self.system.distribution:
            batch_script_content.append(f"#SBATCH --distribution={self.system.distribution}")
        if self.system.gpus_per_node:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")
        if self.system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.system.ntasks_per_node}")
        if "time_limit" in args:
            batch_script_content.append(f"#SBATCH --time={args['time_limit']}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )

    def _format_env_vars(self, env_vars: Dict[str, Any]) -> str:
        """
        Format environment variables for inclusion in a batch script.

        Args:
            env_vars (Dict[str, Any]): Environment variables to format.

        Returns:
            str: A string representation of the formatted environment variables.
        """
        formatted_vars = []
        for key in sorted(env_vars.keys()):
            value = env_vars[key]
            formatted_value = str(value["default"]) if isinstance(value, dict) and "default" in value else str(value)
            formatted_vars.append(f"export {key}={formatted_value}")
        return "\n".join(formatted_vars)
