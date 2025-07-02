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
import os
from pathlib import Path
from typing import Any, Dict, List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .nemo_run import NeMoRunTestDefinition


class NeMoRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo 2.0 on Slurm systems."""

    def image_path(self) -> str | None:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _gen_srun_command(self) -> str:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        self._set_additional_env_vars(tdef)
        return super()._gen_srun_command()

    def _set_additional_env_vars(self, tdef: NeMoRunTestDefinition):
        """Set environment variables based on NeMoRunTestDefinition."""
        self.final_env_vars["CLOUDAI_NEMO_TASK"] = tdef.cmd_args.task
        self.final_env_vars["CLOUDAI_NEMO_RECIPE"] = tdef.cmd_args.recipe_name

        pipeline_model_parallel_size = tdef.cmd_args.trainer.strategy.pipeline_model_parallel_size
        if isinstance(pipeline_model_parallel_size, list):
            pipeline_model_parallel_size = pipeline_model_parallel_size[0]
        pipeline_model_parallel_size = int(pipeline_model_parallel_size)

        if pipeline_model_parallel_size > 1:
            logging.debug("Setting NCCL_P2P_NET_CHUNKSIZE to 2097152 as pipeline_model_parallel_size is greater than 1")
            self.final_env_vars["NCCL_P2P_NET_CHUNKSIZE"] = "2097152"

        enable_fsdp = os.getenv("CLOUDAI_ENABLE_FSDP", "0")
        if enable_fsdp == "1":
            logging.info(
                (
                    "CLOUDAI_ENABLE_FSDP is set to 1. Currently, NemoRun does not support FSDP "
                    "with TP communication overlap."
                )
            )
            self.final_env_vars["CLOUDAI_DISABLE_TP_COMM_OVERLAP"] = "1"

    def _run_script(self) -> Path:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        return tdef.script.installed_path

    def _container_mounts(self) -> List[str]:
        return [f"{self._run_script().parent.absolute()}:/cloudai_workspace"]

    def flatten_dict(self, d: dict[str, str], parent_key: str = "", sep: str = "."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def append_flattened_dict(self, prefix: str, d: Dict[str, Any], command: List[str]):
        flattened = self.flatten_dict(d)
        for key, value in flattened.items():
            if value is not None:
                if prefix:
                    command.append(f"{prefix}.{key}={value}")
                else:
                    command.append(f"{key}={value}")

    def _validate_recipe_name(self, recipe_name: str) -> str:
        """Validate the recipe name against the supported list."""
        supported_recipes = [
            "cloudai_llama3_8b_recipe",
            "cloudai_llama3_70b_recipe",
            "cloudai_llama3_405b_recipe",
            "cloudai_nemotron3_8b_recipe",
            "cloudai_nemotron4_15b_recipe",
            "cloudai_nemotron4_340b_recipe",
        ]

        if recipe_name not in supported_recipes:
            logging.warning(
                f"Using default {recipe_name} in Nemo2.0. "
                "Passing advance CLI options (e.g., factory functions) might not be fully supported in Nemo-Run CLI."
            )

        return recipe_name

    def generate_test_command(self) -> List[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)

        tdef.cmd_args.data.num_train_samples = tdef.update_num_train_samples

        cmd_args_dict = tdef.cmd_args.model_dump()

        for non_cmd_arg in {"docker_image_url", "num_layers", "task", "recipe_name"}:
            cmd_args_dict.pop(non_cmd_arg)

        recipe_name = self._validate_recipe_name(tdef.cmd_args.recipe_name)

        command = ["python", f"/cloudai_install/{self._run_script().name}", "--factory", recipe_name, "-y"]

        num_nodes, _ = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)

        if tdef.cmd_args.trainer.num_nodes is not None and tdef.cmd_args.trainer.num_nodes > num_nodes:
            logging.warning(
                f"Mismatch in num_nodes: real {num_nodes} < requested by test {tdef.cmd_args.trainer.num_nodes}. "
                "cmd_args.trainer.num_nodes value will be overridden to the actual number of nodes."
            )

        cmd_args_dict["trainer"]["num_nodes"] = num_nodes

        if self.system.gpus_per_node:
            trainer_config = cmd_args_dict.get("trainer", {})
            if "devices" in trainer_config:
                user_devices = trainer_config["devices"]
                if user_devices != self.system.gpus_per_node:
                    logging.warning(
                        f"User-specified trainer.devices ({user_devices}) differs from "
                        f"system gpus_per_node ({self.system.gpus_per_node})"
                    )
            cmd_args_dict["trainer"]["devices"] = self.system.gpus_per_node
        else:
            logging.debug("SlurmSystem.gpus_per_node is not set. Skipping trainer.devices injection.")

        self.append_flattened_dict("", cmd_args_dict, command)

        if self.test_run.test.extra_cmd_args:
            command.append(self.test_run.test.extra_cmd_args)

        return command
