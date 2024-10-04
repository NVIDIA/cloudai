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
from typing import Dict, List

from cloudai import CommandGenStrategy

from .kubernetes_install_strategy import NeMoLauncherKubernetesInstallStrategy


class NeMoLauncherKubernetesCommandGenStrategy(CommandGenStrategy):
    """Command generation strategy for NeMo Megatron Launcher on Kubernetes systems."""

    def gen_exec_command(
        self,
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: Path,
        num_nodes: int,
        nodes: List[str],
    ) -> str:
        final_env_vars = {**self.system.global_env_vars, **extra_env_vars}

        launcher_path = (
            self.system.install_path
            / NeMoLauncherKubernetesInstallStrategy.SUBDIR_PATH
            / NeMoLauncherKubernetesInstallStrategy.REPOSITORY_NAME
        )

        self.final_cmd_args = {**self.default_cmd_args, **cmd_args}
        self.final_cmd_args["launcher_scripts_path"] = str(launcher_path / "launcher_scripts")

        self.final_cmd_args.update({f"env_vars.{key}": value for key, value in final_env_vars.items()})

        self.final_cmd_args["cluster"] = self.final_cmd_args.pop("cluster.value", "")
        self.final_cmd_args["training"] = self.final_cmd_args.pop("training.values", "")

        for key in ["repository_url", "repository_commit_hash", "docker_image_url"]:
            self.final_cmd_args.pop(key, None)

        if self.final_cmd_args.get("data_dir") == "DATA_DIR":
            raise ValueError(
                "The 'data_dir' field of the NeMo launcher test contains the placeholder 'DATA_DIR'. "
                "Please update the test schema TOML file with a valid path to the dataset."
            )

        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args)

        full_cmd = f"python {launcher_path}/launcher_scripts/main.py {cmd_args_str}"

        if extra_cmd_args:
            full_cmd += f" {extra_cmd_args}"

        env_vars_str = " ".join(f"{key}={value}" for key, value in final_env_vars.items())
        return f"{env_vars_str} {full_cmd}".strip() if env_vars_str else full_cmd.strip()

    def _generate_cmd_args_str(self, args: Dict[str, str]) -> str:
        """
        Generate a string of command-line arguments, wrapping values in quotes when necessary.

        Args:
            args (Dict[str, str]): The command-line arguments.

        Returns:
            str: A string of command-line arguments.
        """
        cmd_arg_str_parts = []
        env_var_str_parts = []
        special_chars = ["[", "]", "\\"]

        for key, value in args.items():
            if any(char in value for char in special_chars):
                value = f'"{value}"'

            if key.startswith("env_vars."):
                env_var_str_parts.append(f"+{key}={value}")
            else:
                cmd_arg_str_parts.append(f"{key}={value}")

        return " ".join(cmd_arg_str_parts + env_var_str_parts)
