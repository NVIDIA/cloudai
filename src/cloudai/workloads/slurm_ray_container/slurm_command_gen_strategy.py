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

from pathlib import Path
from typing import Any, Dict, List, Union, cast

from jinja2 import Template

from cloudai import TestRun
from cloudai.workloads.slurm_container import SlurmContainerCommandGenStrategy

from .slurm_ray_container import SlurmRayContainerTestDefinition


class SlurmRayContainerCommandGenStrategy(SlurmContainerCommandGenStrategy):
    """Command generation strategy for generic Slurm container tests."""

    def _get_sbatch_directives(self, args: Dict[str, Any], output_path: Path) -> Dict[str, str]:
        sbatch_directives = super()._get_sbatch_directives(args, output_path)
        # TODO(Amey): We probably need to figure out what to do with cpus-per-task, mem-per-cpu
        # override tasks per node
        sbatch_directives["tasks-per-node"] = "1"
        sbatch_directives["exclusive"] = ""

        return sbatch_directives

    def generate_test_command(
        self, env_vars: dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> list[str]:
        tdef: SlurmRayContainerTestDefinition = cast(SlurmRayContainerTestDefinition, tr.test.test_definition)
        srun_command_parts: list[str] = [*super().gen_nsys_command(tr), tdef.cmd_args.cmd]
        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        # load the jinja template file which is placed at the same directory as this file
        script_dir = Path(__file__).parent
        template_path = script_dir / "slurm_ray_container_template.sh.jinja"
        template = Template(template_path.read_text())

        # render the template
        rendered_template = template.render(
            {"conda_env": tdef.cmd_args.conda_env, "command": " ".join(srun_command_parts)}
        )

        return [rendered_template]
