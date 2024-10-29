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


from typing import Any, Dict, List, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.nemo_run import NeMoRunTestDefinition


class NeMoRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo 2.0 on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, str], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def generate_test_command(self, env_vars: Dict[str, str], cmd_args: Dict[str, str], tr: TestRun) -> List[str]:
        command = ["nemo", "llm"]

        task = cmd_args.get("task", "pretrain")
        command.append(task)

        recipe_name = cmd_args.get("recipe_name", "llama3_8b")
        command.extend(["--factory", recipe_name])

        command.append("-y")

        if tr.nodes:
            command.append(f"trainer.num_nodes={len(tr.nodes)}")
        elif tr.num_nodes > 0:
            command.append(f"trainer.num_nodes={tr.num_nodes}")

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
