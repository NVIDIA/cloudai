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


import logging
import sys
from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.nemo_run import NeMoRunTestDefinition


class NeMoRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo 2.0 on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _container_mounts(self, tr: TestRun) -> List[str]:
        return []

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

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)

        cmd_args_dict = tdef.cmd_args.model_dump()

        cmd_args_dict.pop("docker_image_url")
        cmd_args_dict.pop("num_layers")

        command = ["nemo", "llm", cmd_args_dict.pop("task"), "--factory", cmd_args_dict.pop("recipe_name"), "-y"]

        num_nodes = len(self.system.parse_nodes(tr.nodes)) if tr.nodes else tr.num_nodes

        if cmd_args_dict["trainer"]["num_nodes"] and cmd_args_dict["trainer"]["num_nodes"] > num_nodes:
            err = (
                f"Mismatch in num_nodes: {num_nodes} vs {cmd_args_dict['trainer']['num_nodes']}. "
                "trainer.num_nodes should be less than or equal to the number of nodes specified "
                "in the test scenario."
            )

            logging.warning(err)
            sys.exit(1)

        cmd_args_dict["trainer"]["num_nodes"] = num_nodes

        self.append_flattened_dict("", cmd_args_dict, command)

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
