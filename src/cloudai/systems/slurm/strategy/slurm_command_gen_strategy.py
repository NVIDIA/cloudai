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

from cloudai import CommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.util.docker_image_cache_manager import DockerImageCacheManager


class SlurmCommandGenStrategy(CommandGenStrategy):
    """
    Abstract base class for defining command generation strategies specific to Slurm environments.

    Attributes
        slurm_system (SlurmSystem): A casted version of the `system` attribute, which provides Slurm-specific
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
        self.slurm_system = system
        if not self.slurm_system.default_partition:
            raise ValueError(
                "Default partition not set in the Slurm system object. "
                "The 'default_partition' attribute should be properly defined in the Slurm system configuration. "
                "Please ensure that 'default_partition' is set correctly in the corresponding system configuration "
                "(e.g., system.toml)."
            )

        self.install_path = self.slurm_system.install_path

        self.docker_image_cache_manager = DockerImageCacheManager(
            self.slurm_system.install_path,
            self.slurm_system.cache_docker_images_locally,
            self.slurm_system.default_partition,
        )
        self.docker_image_url = self.cmd_args.get("docker_image_url", "")

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

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        """
        Parse command arguments to configure Slurm job settings.

        Args:
            job_name_prefix (str): Prefix for the job name.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            num_nodes (int): The number of nodes to be used for the test execution.
            nodes (List[str]): List of nodes for the job.

        Returns:
            Dict[str, Any]: Dictionary containing configuration for Slurm job.

        Raises:
            KeyError: If partition or essential node settings are missing.
        """
        job_name = self.job_name(job_name_prefix)

        parsed_nodes = self.slurm_system.parse_nodes(nodes)
        num_nodes = len(parsed_nodes) if parsed_nodes else num_nodes
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
        if self.slurm_system.account:
            job_name = f"{self.slurm_system.account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return job_name

    def generate_full_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> str:
        srun_command_parts = self.generate_srun_command(slurm_args, env_vars, cmd_args, extra_cmd_args)
        test_command_parts = self.generate_test_command(slurm_args, env_vars, cmd_args, extra_cmd_args)
        return " \\\n".join(srun_command_parts + test_command_parts)

    def generate_srun_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        srun_command_parts = ["srun", f"--mpi={self.slurm_system.mpi}"]
        if slurm_args.get("image_path"):
            srun_command_parts.append(f'--container-image={slurm_args["image_path"]}')
            if slurm_args.get("container_mounts"):
                srun_command_parts.append(f'--container-mounts={slurm_args["container_mounts"]}')

        if self.slurm_system.extra_srun_args:
            srun_command_parts.append(self.slurm_system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
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
        if self.slurm_system.extra_srun_args and reservation_key in self.slurm_system.extra_srun_args:
            reservation = self.slurm_system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            batch_script_content.append(f"#SBATCH --reservation={reservation}")

        return batch_script_content

    def _write_sbatch_script(
        self, args: Dict[str, Any], env_vars_str: str, srun_command: str, output_path: Path
    ) -> str:
        """
        Write the batch script for Slurm submission and return the sbatch command.

        Args:
            args (Dict[str, Any]): Arguments including job settings.
            env_vars_str (str): Environment variables.
            srun_command (str): srun command.
            output_path (Path): Output directory for script and logs.

        Returns:
            str: sbatch command to submit the job.
        """
        batch_script_content = [
            "#!/bin/bash",
            f"#SBATCH --job-name={args['job_name']}",
            f"#SBATCH -N {args['num_nodes']}",
        ]

        self._append_sbatch_directives(batch_script_content, args, output_path)

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
        batch_script_content.append(f"#SBATCH --partition={self.slurm_system.default_partition}")
        if args["node_list_str"]:
            batch_script_content.append(f"#SBATCH --nodelist={args['node_list_str']}")
        if self.slurm_system.account:
            batch_script_content.append(f"#SBATCH --account={self.slurm_system.account}")
        if self.slurm_system.distribution:
            batch_script_content.append(f"#SBATCH --distribution={self.slurm_system.distribution}")
        if self.slurm_system.gpus_per_node:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.slurm_system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.slurm_system.gpus_per_node}")
        if self.slurm_system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.slurm_system.ntasks_per_node}")
        if "time_limit" in args:
            batch_script_content.append(f"#SBATCH --time={args['time_limit']}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )
