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

from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import NeMoLauncherSlurmInstallStrategy


class NeMoLauncherSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo Megatron Launcher on Slurm systems."""

    def gen_exec_command(
        self,
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        num_nodes: int,
        nodes: List[str],
    ) -> str:
        final_env_vars = self._override_env_vars(self.system.global_env_vars, extra_env_vars)

        launcher_path = (
            self.system.install_path
            / NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH
            / NeMoLauncherSlurmInstallStrategy.REPOSITORY_NAME
        )
        overriden_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        self.final_cmd_args = {
            k: self._handle_special_keys(k, v, str(launcher_path), str(output_path))
            for k, v in overriden_cmd_args.items()
        }
        self.final_cmd_args["base_results_dir"] = str(output_path)
        self.final_cmd_args["training.model.data.index_mapping_dir"] = str(output_path)
        self.final_cmd_args["launcher_scripts_path"] = str(launcher_path / "launcher_scripts")

        for key, value in final_env_vars.items():
            self.final_cmd_args[f"env_vars.{key}"] = value

        if "training.values" in self.final_cmd_args:
            self.final_cmd_args["training"] = self.final_cmd_args.pop("training.values")

        self.final_cmd_args["cluster.partition"] = self.slurm_system.default_partition
        reservation_key = "--reservation "
        if self.slurm_system.extra_srun_args and reservation_key in self.slurm_system.extra_srun_args:
            reservation = self.slurm_system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            self.final_cmd_args["+cluster.reservation"] = reservation

        nodes = self.slurm_system.parse_nodes(nodes)
        if nodes:
            self.final_cmd_args["training.trainer.num_nodes"] = str(len(nodes))
        else:
            self.final_cmd_args["training.trainer.num_nodes"] = num_nodes

        self.final_cmd_args["container"] = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH,
            NeMoLauncherSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        del self.final_cmd_args["repository_url"]
        del self.final_cmd_args["repository_commit_hash"]
        del self.final_cmd_args["docker_image_url"]

        if self.slurm_system.account is not None:
            self.final_cmd_args["cluster.account"] = self.slurm_system.account
            self.final_cmd_args["cluster.job_name_prefix"] = f"{self.slurm_system.account}-cloudai.nemo:"
        self.final_cmd_args["cluster.gpus_per_node"] = (
            self.slurm_system.gpus_per_node if self.slurm_system.gpus_per_node is not None else "null"
        )

        if ("data_dir" in self.final_cmd_args) and (self.final_cmd_args["data_dir"] == "DATA_DIR"):
            raise ValueError(
                "The 'data_dir' field of the NeMo launcher test contains the placeholder 'DATA_DIR'. "
                "Please update the test schema TOML file with a valid path to the dataset. "
                "The 'data_dir' field must point to an actual dataset location, not a placeholder."
            )

        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args, nodes)

        full_cmd = f"python {launcher_path}/launcher_scripts/main.py {cmd_args_str}"

        if extra_cmd_args:
            full_cmd += f" {extra_cmd_args}"
            tokenizer_key = "training.model.tokenizer.model="
            if tokenizer_key in extra_cmd_args:
                tokenizer_path = extra_cmd_args.split(tokenizer_key, 1)[1].split(" ", 1)[0]
                if not Path(tokenizer_path).is_file():
                    raise ValueError(
                        f"The provided tokenizer path '{tokenizer_path}' is not valid. "
                        "Please review the test schema file to ensure the tokenizer path is correct. "
                        "If it contains a placeholder value, refer to USER_GUIDE.md to download the tokenizer "
                        "and update the schema file accordingly."
                    )
                full_cmd += f" container_mounts=[{tokenizer_path}:{tokenizer_path}]"

        env_vars_str = " ".join(f"{key}={value}" for key, value in final_env_vars.items())
        full_cmd = f"{env_vars_str} {full_cmd}" if env_vars_str else full_cmd

        # Log the generated command to a bash file
        self._log_command_to_file(full_cmd, output_path)

        return full_cmd.strip()

    def _handle_special_keys(self, key: str, value: Any, launcher_path: str, output_path: str) -> Any:
        """
        Handle special formatting for specific keys.

        Args:
            key (str): The argument key.
            value (Any): The argument value.
            launcher_path (str): The base path for NeMo Megatron launcher.
            output_path (str): Path to the output directory.

        Returns:
            Any: The specially formatted value, if applicable.
        """
        if key == "training.model.data.data_prefix":
            return value.replace("\\", "")

        return value

    def _generate_cmd_args_str(self, args: Dict[str, str], nodes: List[str]) -> str:
        """
        Generate a string of command-line arguments.

        Args:
            args (Dict[str, str]): The command-line arguments.
            nodes (List[str]): A list of nodes where the test will be executed.

        Returns:
            str: A string of command-line arguments.
        """
        cmd_arg_str_parts = []
        env_var_str_parts = []

        for key, value in args.items():
            if key.startswith("env_vars."):
                if isinstance(value, str) and "," in value:
                    value = f"\\'{value}\\'"
                env_var_str_parts.append(f"+{key}={value}")
            else:
                cmd_arg_str_parts.append(f"{key}={value}")

        if nodes:
            nodes_str = ",".join(nodes)
            cmd_arg_str_parts.append(f"+cluster.nodelist=\\'{nodes_str}\\'\n")

        return " ".join(cmd_arg_str_parts + env_var_str_parts)

    def _log_command_to_file(self, command: str, output_path: Path):
        """Log the generated command to a bash file in the specified output path."""
        log_file = output_path / "generated_command.sh"

        # Ensure the output path exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        command_with_line_breaks = command.replace(" ", " \\\n ")

        with log_file.open("a") as f:
            f.write(f"{command_with_line_breaks}\n\n")
