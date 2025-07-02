# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .nemo_launcher import NeMoLauncherTestDefinition


class NeMoLauncherSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo Megatron Launcher on Slurm systems."""

    job_prefix: Optional[str] = None

    def _container_mounts(self) -> list[str]:
        # this strategy handles container mounts in a different way, so it is OK to return an empty list
        return []

    def gen_exec_command(self) -> str:
        self._prepare_environment()

        _, nodes = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
        self._set_node_config(nodes, self.test_run.nnodes)

        tdef: NeMoLauncherTestDefinition = cast(NeMoLauncherTestDefinition, self.test_run.test.test_definition)

        if self.system.account:
            self.final_cmd_args["cluster.account"] = self.system.account

        self.final_cmd_args["container"] = str(tdef.docker_image.installed_path)
        self.final_cmd_args.pop("docker_image_url", None)

        if self.job_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.system.account:
                self.job_prefix = f"{self.system.account}-cloudai.nemo_{timestamp}"
            else:
                self.job_prefix = timestamp

        self.final_cmd_args["cluster.job_name_prefix"] = f"{self.job_prefix}:"

        self.final_cmd_args["cluster.gpus_per_node"] = self.system.gpus_per_node or "null"

        repo_path = (
            tdef.python_executable.git_repo.installed_path.absolute()
            if tdef.python_executable.git_repo.installed_path is not None
            else None
        )
        if not repo_path:
            logging.warning(
                f"Local clone of git repo {tdef.python_executable.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            repo_path = self.system.install_path / tdef.python_executable.git_repo.repo_name  # dry-run compatibility
        venv_path = tdef.python_executable.venv_path
        if not venv_path:
            logging.warning(
                f"The virtual environment for git repo {tdef.python_executable.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            venv_path = self.system.install_path / tdef.python_executable.venv_name  # dry-run compatibility
        py_bin = (venv_path / "bin" / "python").absolute()
        self.final_cmd_args.update(
            {
                "base_results_dir": str(self.test_run.output_path.absolute()),
                "launcher_scripts_path": str((repo_path / tdef.cmd_args.launcher_script).parent.absolute()),
            }
        )

        nccl_topo_file = self.final_env_vars.get("NCCL_TOPO_FILE")
        if nccl_topo_file:
            self.final_cmd_args["container_mounts"] = f'["{nccl_topo_file}:{nccl_topo_file}"]'

        self._validate_data_config()

        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args, nodes)
        full_cmd = f"{py_bin} \\\n {repo_path / tdef.cmd_args.launcher_script} \\\n {cmd_args_str}"
        if self.test_run.test.extra_cmd_args:
            full_cmd += f" {self.test_run.test.extra_cmd_args}"
        full_cmd = self._update_container_mounts_with_tokenizer_path(full_cmd)

        env_vars_str = self._gen_env_vars_str(self.final_env_vars)
        full_cmd = f"{env_vars_str}{full_cmd}" if env_vars_str else full_cmd

        # Log the generated command to a bash file
        self._log_command_to_file(full_cmd, self.test_run.output_path)

        return full_cmd.strip()

    def _gen_env_vars_str(self, env: dict[str, str | list[str]]) -> str:
        env_vars_str = " \\\n".join(f'{key}="{value}"' for key, value in env.items())
        if env_vars_str:
            env_vars_str += " \\\n"
        return env_vars_str

    def _prepare_environment(self) -> None:
        """
        Prepare the environment variables and command arguments.

        Args:
            cmd_args (Dict[str, Union[str, List[str]]]): Command-line arguments for the launcher.
            extra_env_vars (Dict[str, Union[str, List[str]]]): Additional environment variables.
            output_path (Path): Path to the output directory.
        """
        overriden_cmd_args = self._flatten_dict(self.test_run.test.cmd_args)
        overriden_cmd_args.pop("launcher_script", None)
        self.final_cmd_args = {k: self._handle_special_keys(k, v) for k, v in sorted(overriden_cmd_args.items())}

        for key, value in self.final_env_vars.items():
            self.final_cmd_args[f"env_vars.{key}"] = value

        if "training.values" in self.final_cmd_args:
            self.final_cmd_args["training"] = self.final_cmd_args.pop("training.values")

        self.final_cmd_args["cluster.partition"] = self.system.default_partition
        self._handle_reservation()

    def _handle_reservation(self) -> None:
        """Handle Slurm reservation if provided."""
        reservation_key = "--reservation "
        if self.system.extra_srun_args and reservation_key in self.system.extra_srun_args:
            reservation = self.system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            self.final_cmd_args["+cluster.reservation"] = reservation

    def _set_node_config(self, nodes: List[str], num_nodes: int) -> None:
        """
        Set the number of nodes configuration.

        Args:
            nodes (List[str]): List of nodes where the test will run.
            num_nodes (int): Number of nodes to allocate if no specific node list is provided.
        """
        self.final_cmd_args["training.trainer.num_nodes"] = str(len(nodes)) if nodes else num_nodes

    def _validate_data_config(self) -> None:
        """Validate the data prefix configuration for non-mock environments."""
        if self.final_cmd_args.get("training.model.data.data_impl") != "mock":
            data_prefix = self.final_cmd_args.get("training.model.data.data_prefix")
            if data_prefix == "[]":
                raise ValueError(
                    "The 'data_prefix' field of the NeMo launcher test is missing or empty. "
                    "Please update the test schema TOML file with a valid prefix for the training datasets."
                )

    def _update_container_mounts_with_tokenizer_path(self, full_cmd: str) -> str:
        tokenizer_key = "training.model.tokenizer.model="
        if tokenizer_key in full_cmd:
            tokenizer_path = full_cmd.split(tokenizer_key, 1)[1].split(" ", 1)[0]
            if not Path(tokenizer_path).is_file():
                raise ValueError(
                    f"The provided tokenizer path '{tokenizer_path}' is not valid. "
                    "Please review the test schema file to ensure the tokenizer path is correct. "
                    "If it contains a placeholder value, refer to USER_GUIDE.md to download the tokenizer "
                    "and update the schema file accordingly."
                )

            container_mounts_entry = f'"{tokenizer_path}:{tokenizer_path}"'
            if "container_mounts=" in full_cmd:
                full_cmd = full_cmd.replace("container_mounts=[", f"container_mounts=[{container_mounts_entry}, ")
            else:
                full_cmd += f" container_mounts=[{container_mounts_entry}]"

        return full_cmd

    def _generate_cmd_args_str(self, args: Dict[str, Union[str, List[str]]], nodes: List[str]) -> str:
        """
        Generate a string of command-line arguments.

        Args:
            args (Dict[str, Union[str, List[str]]]): The command-line arguments.
            nodes (List[str]): A list of nodes where the test will be executed.

        Returns:
            str: A string of command-line arguments.
        """
        cmd_arg_str_parts = []
        env_var_str_parts = []

        for key, value in args.items():
            if key.startswith("env_vars."):
                if isinstance(value, str) and "," in value:
                    env_var_str_parts.append(f"+{key}=\\'{value}\\'")
                else:
                    env_var_str_parts.append(f'+{key}="{value}"')
            else:
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                cmd_arg_str_parts.append(f"{key}={value}")

        if nodes:
            nodes_str = ",".join(nodes)
            cmd_arg_str_parts.append(f"+cluster.nodelist=\\'{nodes_str}\\'")

        return " \\\n ".join(cmd_arg_str_parts + env_var_str_parts)

    def _log_command_to_file(self, command: str, output_path: Path):
        """Log the generated command to a bash file in the specified output path."""
        log_file = output_path / "generated_command.sh"

        # Ensure the output path exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("a") as f:
            f.write(f"{command}\n\n")

    def _handle_special_keys(self, key: str, value: Any) -> Any:
        """
        Handle special formatting for specific keys.

        Args:
            key (str): The argument key.
            value (Any): The argument value.

        Returns:
            Any: The specially formatted value, if applicable.
        """
        if key == "training.model.data.data_prefix":
            return value.replace("\\", "")

        return value
