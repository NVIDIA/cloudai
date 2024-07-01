#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import NeMoLauncherSlurmInstallStrategy

REQUIRE_ENV_VARS = [
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_QPS_PER_CONNECTION",
    "UCX_IB_GID_INDEX",
    "NCCL_IB_ADAPTIVE_ROUTING",
    "NCCL_IB_SPLIT_DATA_ON_QPS",
    "NCCL_IBEXT_DISABLE",
]


class NeMoLauncherSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """
    Command generation strategy for NeMo Megatron Launcher on Slurm systems.

    Attributes
        install_path (str): The installation path of CloudAI.
    """

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
        # Ensure required environment variables are included
        for key in REQUIRE_ENV_VARS:
            if key not in extra_env_vars:
                extra_env_vars[key] = self.slurm_system.global_env_vars[key]

        launcher_path = os.path.join(
            self.install_path,
            NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH,
            NeMoLauncherSlurmInstallStrategy.REPOSITORY_NAME,
        )
        overriden_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        self.final_cmd_args = {
            k: self._handle_special_keys(k, v, launcher_path, output_path) for k, v in overriden_cmd_args.items()
        }
        self.final_cmd_args["base_results_dir"] = output_path
        self.final_cmd_args["training.model.data.index_mapping_dir"] = output_path
        self.final_cmd_args["launcher_scripts_path"] = os.path.join(launcher_path, "launcher_scripts")
        for key, value in extra_env_vars.items():
            self.final_cmd_args[f"env_vars.{key}"] = value
        self.final_cmd_args["cluster.partition"] = self.slurm_system.default_partition
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

        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args, nodes)

        full_cmd = f"./venv/bin/python {launcher_path}/launcher_scripts/main.py {cmd_args_str}"

        if extra_cmd_args:
            full_cmd += " " + extra_cmd_args
            if "training.model.tokenizer.model" in extra_cmd_args:
                tokenizer_path = extra_cmd_args.split("training.model.tokenizer.model=")[1].split(" ")[0]
                full_cmd += " " + f"container_mounts=[{tokenizer_path}:{tokenizer_path}]"

        env_vars_str = " ".join(f"{key}={value}" for key, value in extra_env_vars.items())
        full_cmd = f"{env_vars_str} {full_cmd}" if env_vars_str else full_cmd

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
        arg_str_parts = []
        for key, value in args.items():
            formatted_key = f"+{key}" if key.startswith("env_vars.") else key
            arg_str_parts.append(f"{formatted_key}={value}")

        if nodes:
            nodes_str = ",".join(nodes)
            arg_str_parts.append(f"+cluster.nodelist=\\'{nodes_str}\\'")

        return " ".join(arg_str_parts)
