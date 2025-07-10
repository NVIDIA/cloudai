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


from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .megatron_run import MegatronRunTestDefinition


class MegatronRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for MegatronRun on Slurm systems."""

    def image_path(self) -> str | None:
        tdef: MegatronRunTestDefinition = cast(MegatronRunTestDefinition, self.test_run.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _container_mounts(self) -> list[str]:
        return []

    def generate_test_command(self) -> list[str]:
        tdef: MegatronRunTestDefinition = cast(MegatronRunTestDefinition, self.test_run.test.test_definition)

        command = [
            "python",
            str((tdef.cmd_args.run_script).absolute()),
            *[f"{k} {v}" for k, v in tdef.cmd_args_dict.items()],
        ]

        if self.test_run.test.extra_cmd_args:
            command.append(self.test_run.test.extra_cmd_args)

        return command
