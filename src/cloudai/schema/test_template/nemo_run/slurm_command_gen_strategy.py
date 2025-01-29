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


from typing import Any, Dict, List, Union, cast

from pydantic import BaseModel

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

    def flatten_dict(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, BaseModel):
                items.extend(self.flatten_dict(v.model_dump(), new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)

        command = ["nemo", "llm", tdef.cmd_args.task, "--factory", tdef.cmd_args.recipe_name, "-y"]

        if tr.nodes:
            command.append(f"trainer.num_nodes={len(self.system.parse_nodes(tr.nodes))}")
        elif tr.num_nodes > 0:
            command.append(f"trainer.num_nodes={tr.num_nodes}")

        if hasattr(tdef.cmd_args, "trainer"):
            for key, value in tdef.cmd_args.trainer.__dict__.items():
                command.append(f"trainer.{key}={value}")

        if hasattr(tdef.cmd_args, "log"):
            flattened_log = self.flatten_dict(tdef.cmd_args.log.__dict__)
            for key, value in flattened_log.items():
                command.append(f"log.{key}={value}")

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
