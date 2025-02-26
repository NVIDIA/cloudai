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
from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .megatron_run import MegatronRunTestDefinition


class MegatronRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for MegatronRun on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: MegatronRunTestDefinition = cast(MegatronRunTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _container_mounts(self, tr: TestRun) -> List[str]:
        tdef: MegatronRunTestDefinition = cast(MegatronRunTestDefinition, tr.test.test_definition)
        mounts: list[str] = [f"{tdef.cmd_args.run_script.parent}"]

        if tdef.cmd_args.save:
            mounts.append(f"{tdef.cmd_args.save}")
        if tdef.cmd_args.load and str(tdef.cmd_args.load) not in mounts:
            mounts.append(f"{tdef.cmd_args.load}")

        if tdef.cmd_args.tokenizer_model:
            mounts.append(f"{tdef.cmd_args.tokenizer_model.parent}")

        return mounts

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: MegatronRunTestDefinition = cast(MegatronRunTestDefinition, tr.test.test_definition)

        command = [
            "python",
            str((tdef.cmd_args.run_script).absolute()),
            *[f"{k} {v}" for k, v in tdef.cmd_args_dict.items()],
        ]

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
