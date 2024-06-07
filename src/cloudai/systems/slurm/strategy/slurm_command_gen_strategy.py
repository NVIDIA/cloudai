# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
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

    def __init__(
        self,
        system: SlurmSystem,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        """
        Initialize a new SlurmCommandGenStrategy instance.

        Args:
            system (System): The system schema object.
            env_vars (Dict[str, Any]): Environment variables.
            cmd_args (Dict[str, Any]): Command-line arguments.
        """
        super().__init__(system, env_vars, cmd_args)
        self.slurm_system = system
        self.install_path = self.slurm_system.install_path
        self.default_env_vars.update(self.slurm_system.global_env_vars)

        self.docker_image_cache_manager = DockerImageCacheManager(
            self.slurm_system.install_path,
            self.slurm_system.cache_docker_images_locally,
            self.slurm_system.default_partition,
        )
        docker_image_url_info = self.cmd_args.get("docker_image_url")
        if docker_image_url_info is not None:
            self.docker_image_url = docker_image_url_info.get("default")
        else:
            self.docker_image_url = ""

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
        account = self.slurm_system.account
        if account is None:
            job_name = f"{job_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            job_name = f"{account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        partition = self.slurm_system.default_partition
        if not partition:
            raise KeyError("Partition not specified in the system configuration.")

        parsed_nodes = self.slurm_system.parse_nodes(nodes)
        num_nodes = len(parsed_nodes) if parsed_nodes else num_nodes
        node_list_str = ",".join(parsed_nodes) if parsed_nodes else ""

        slurm_args = {
            "job_name": job_name,
            "partition": partition,
            "num_nodes": num_nodes,
            "node_list_str": node_list_str,
        }
        if self.slurm_system.account:
            slurm_args["account"] = self.slurm_system.account
        if self.slurm_system.distribution:
            slurm_args["distribution"] = self.slurm_system.distribution
        if self.slurm_system.gpus_per_node:
            slurm_args["gpus_per_node"] = self.slurm_system.gpus_per_node
        if self.slurm_system.ntasks_per_node:
            slurm_args["ntasks_per_node"] = self.slurm_system.ntasks_per_node
        if "time_limit" in cmd_args:
            slurm_args["time_limit"] = cmd_args["time_limit"]

        return slurm_args

    def _generate_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        """
        Generate the srun command string for executing the test.

        Args:
            slurm_args (Dict[str, Any]): Arguments containing Slurm job settings including image path and container
                mounts.
            env_vars (Dict[str, str]): Environment variables.
            cmd_args (Dict[str, str]): Command-line arguments.
            extra_cmd_args (str): Additional command-line arguments to be included in the srun command.

        Returns:
            str: The complete srun command to execute the test.
        """
        return ""

    def _write_sbatch_script(self, args: Dict[str, Any], env_vars_str: str, srun_command: str, output_path: str) -> str:
        """
        Write the batch script for Slurm submission and returns the sbatch command.

        Args:
            args (Dict[str, Any]): Arguments including job settings.
            env_vars_str (str): Environment variables.
            srun_command (str): srun command.
            output_path (str): Output directory for script and logs.

        Returns:
            str: sbatch command to submit the job.
        """
        batch_script_content = [
            "#!/bin/bash",
            f"#SBATCH --job-name={args['job_name']}",
            f"#SBATCH -N {args['num_nodes']}",
        ]

        if "output" not in args:
            batch_script_content.append(f"#SBATCH --output={os.path.join(output_path, 'stdout.txt')}")
        if "error" not in args:
            batch_script_content.append(f"#SBATCH --error={os.path.join(output_path, 'stderr.txt')}")
        if args["partition"]:
            batch_script_content.append(f"#SBATCH --partition={args['partition']}")
        if args["node_list_str"]:
            batch_script_content.append(f"#SBATCH --nodelist={args['node_list_str']}")
        if "account" in args:
            batch_script_content.append(f"#SBATCH --account={args['account']}")
        if "distribution" in args:
            batch_script_content.append(f"#SBATCH --distribution={args['distribution']}")
        if "gpus_per_node" in args:
            batch_script_content.append(f"#SBATCH --gpus-per-node={args['gpus_per_node']}")
        if "ntasks_per_node" in args:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={args['ntasks_per_node']}")
        if "time_limit" in args:
            batch_script_content.append(f"#SBATCH --time={args['time_limit']}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )

        batch_script_content.extend(["", env_vars_str, "", srun_command])

        batch_script_path = os.path.join(output_path, "cloudai_sbatch_script.sh")
        with open(batch_script_path, "w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"
