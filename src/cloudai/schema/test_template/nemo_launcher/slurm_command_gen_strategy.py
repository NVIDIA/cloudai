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
        self._prepare_environment(cmd_args, extra_env_vars, output_path)

        nodes = self.slurm_system.parse_nodes(nodes)
        self._set_node_config(nodes, num_nodes)

        self.final_cmd_args["container"] = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH,
            NeMoLauncherSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        for key in ("repository_url", "repository_commit_hash", "docker_image_url"):
            self.final_cmd_args.pop(key, None)

        if self.slurm_system.account:
            self.final_cmd_args.update(
                {
                    "cluster.account": self.slurm_system.account,
                    "cluster.job_name_prefix": f"{self.slurm_system.account}-cloudai.nemo:",
                }
            )
        self.final_cmd_args["cluster.gpus_per_node"] = self.slurm_system.gpus_per_node or "null"

        self._validate_data_config()

        if self.final_cmd_args.get("training.model.data.data_impl") == "mock":
            self.final_cmd_args.pop("data_dir", None)

        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args, nodes)

        full_cmd = f"python {self._launcher_scripts_path()}/launcher_scripts/main.py {cmd_args_str}"

        if extra_cmd_args:
            full_cmd = self._handle_extra_cmd_args(full_cmd, extra_cmd_args)

        env_vars_str = " ".join(f"{key}={value}" for key, value in self.final_env_vars.items())
        full_cmd = f"{env_vars_str} {full_cmd}" if env_vars_str else full_cmd

        return full_cmd.strip()

    def _prepare_environment(self, cmd_args: Dict[str, str], extra_env_vars: Dict[str, str], output_path: Path) -> None:
        """
        Prepare the environment variables and command arguments.

        Args:
            cmd_args (Dict[str, str]): Command-line arguments for the launcher.
            extra_env_vars (Dict[str, str]): Additional environment variables.
            output_path (Path): Path to the output directory.
        """
        self.final_env_vars = self._override_env_vars(self.system.global_env_vars, extra_env_vars)

        launcher_path = self._launcher_scripts_path()
        output_path_abs = output_path.absolute()
        overriden_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        self.final_cmd_args = {
            k: self._handle_special_keys(k, v, str(launcher_path), str(output_path_abs))
            for k, v in overriden_cmd_args.items()
        }
        self.final_cmd_args.update(
            {
                "base_results_dir": str(output_path_abs),
                "training.model.data.index_mapping_dir": str(output_path_abs),
                "launcher_scripts_path": str(launcher_path / "launcher_scripts"),
            }
        )

        for key, value in self.final_env_vars.items():
            self.final_cmd_args[f"env_vars.{key}"] = value

        if "training.values" in self.final_cmd_args:
            self.final_cmd_args["training"] = self.final_cmd_args.pop("training.values")

        self.final_cmd_args["cluster.partition"] = self.slurm_system.default_partition
        self._handle_reservation()

    def _handle_reservation(self) -> None:
        """Handle Slurm reservation if provided."""
        reservation_key = "--reservation "
        if self.slurm_system.extra_srun_args and reservation_key in self.slurm_system.extra_srun_args:
            reservation = self.slurm_system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            self.final_cmd_args["+cluster.reservation"] = reservation

    def _launcher_scripts_path(self) -> Path:
        """
        Return the launcher scripts path.

        Returns
            Path: Absolute path to the NeMo launcher scripts directory.
        """
        return (
            self.system.install_path
            / NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH
            / NeMoLauncherSlurmInstallStrategy.REPOSITORY_NAME
        ).absolute()


        # Log the generated command to a bash file
        self._log_command_to_file(full_cmd, output_path)

        return full_cmd.strip()

    def _set_node_config(self, nodes: List[str], num_nodes: int) -> None:
        """
        Set the number of nodes configuration.

        Args:
            nodes (List[str]): List of nodes where the test will run.
            num_nodes (int): Number of nodes to allocate if no specific node list is provided.
        """
        self.final_cmd_args["training.trainer.num_nodes"] = str(len(nodes)) if nodes else num_nodes

    def _validate_data_config(self) -> None:
        """Validate the data directory and prefix configuration for non-mock environments."""
        if self.final_cmd_args.get("training.model.data.data_impl") != "mock":
            data_dir = self.final_cmd_args.get("data_dir")
            data_prefix = self.final_cmd_args.get("training.model.data.data_prefix")

            if not data_dir or data_dir == "~":
                raise ValueError(
                    "The 'data_dir' field of the NeMo launcher test contains an invalid placeholder '~'. "
                    "Please provide a valid path to the dataset in the test schema TOML file. "
                    "The 'data_dir' field must point to an actual dataset location."
                )

            if data_prefix == "[]":
                raise ValueError(
                    "The 'data_prefix' field of the NeMo launcher test is missing or empty. "
                    "Please update the test schema TOML file with a valid prefix for the training datasets."
                )

    def _handle_extra_cmd_args(self, full_cmd: str, extra_cmd_args: str) -> str:
        """
        Handle additional command arguments such as the tokenizer path.

        Args:
            full_cmd (str): The full command string generated so far.
            extra_cmd_args (str): Additional command-line arguments to append.

        Returns:
            str: Updated command string with the additional arguments.
        """
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
        return full_cmd

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
                if value == "~":
                    cmd_arg_str_parts.append(f"~{key}=null")
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

