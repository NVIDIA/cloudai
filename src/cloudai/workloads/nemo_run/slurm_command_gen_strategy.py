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
import sys
from pathlib import Path
from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.nemo_run import NeMoRunTestDefinition


class NeMoRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo 2.0 on Slurm systems."""

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        self._set_additional_env_vars(env_vars, tdef)

        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _set_additional_env_vars(self, env_vars: Dict[str, Union[str, List[str]]], tdef: NeMoRunTestDefinition):
        """Set environment variables based on NeMoRunTestDefinition."""
        env_vars["CLOUDAI_NEMO_TASK"] = tdef.cmd_args.task
        env_vars["CLOUDAI_NEMO_RECIPE"] = tdef.cmd_args.recipe_name

        pipeline_model_parallel_size = tdef.cmd_args.trainer.strategy.pipeline_model_parallel_size
        if isinstance(pipeline_model_parallel_size, list):
            pipeline_model_parallel_size = pipeline_model_parallel_size[0]
        pipeline_model_parallel_size = int(pipeline_model_parallel_size)

        if pipeline_model_parallel_size > 1:
            logging.debug("Setting NCCL_P2P_NET_CHUNKSIZE to 2097152 as pipeline_model_parallel_size is greater than 1")
            env_vars["NCCL_P2P_NET_CHUNKSIZE"] = "2097152"

    def _run_script(self, tr: TestRun) -> Path:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        return tdef.script.installed_path

    def _container_mounts(self, tr: TestRun) -> List[str]:
        return [f"{self._run_script(tr).parent.absolute()}:/cloudai_workspace"]

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

    def update_num_train_samples(self, tr: TestRun) -> None:
        """Update num_train_samples based on global_batch_size and max_steps."""
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)

        global_batch_size = (
            tdef.cmd_args.data.global_batch_size[0]
            if isinstance(tdef.cmd_args.data.global_batch_size, list)
            else tdef.cmd_args.data.global_batch_size
        )

        max_steps = (
            tdef.cmd_args.trainer.max_steps[0]
            if isinstance(tdef.cmd_args.trainer.max_steps, list)
            else tdef.cmd_args.trainer.max_steps
        )

        tdef.cmd_args.data.num_train_samples = global_batch_size * max_steps

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)

        self.update_num_train_samples(tr)

        cmd_args_dict = tdef.cmd_args.model_dump()

        for non_cmd_arg in {"docker_image_url", "num_layers", "task", "recipe_name"}:
            cmd_args_dict.pop(non_cmd_arg)

        recipe_name = self._validate_recipe_name(tdef.cmd_args.recipe_name)

        command = [
            "python",
            f"/cloudai_install/{self._run_script(tr).name}",
            "--factory",
            recipe_name,
            "-y",
        ]

        num_nodes, _ = self.system.get_nodes_by_spec(tr.nnodes, tr.nodes)

        if cmd_args_dict["trainer"]["num_nodes"] and cmd_args_dict["trainer"]["num_nodes"] > num_nodes:
            err = (
                f"Mismatch in num_nodes: {num_nodes} vs {cmd_args_dict['trainer']['num_nodes']}. "
                "trainer.num_nodes should be less than or equal to the number of nodes specified "
                "in the test scenario."
            )

            logging.error(err)
            sys.exit(1)

        cmd_args_dict["trainer"]["num_nodes"] = num_nodes

        self.append_flattened_dict("", cmd_args_dict, command)

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
