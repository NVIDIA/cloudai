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

from typing import Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_container import SlurmContainerTestDefinition


class SlurmContainerCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for generic Slurm container tests."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def gen_nsys_command(self, tr: TestRun) -> list[str]:
        """NSYS command is generated as part of the test command and disabled here."""
        return []

    def image_path(self, tr: TestRun) -> str | None:
        tdef: SlurmContainerTestDefinition = cast(SlurmContainerTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def gen_srun_prefix(self, tr: TestRun, use_pretest_extras: bool = False) -> list[str]:
        cmd = super().gen_srun_prefix(tr)
        tdef: SlurmContainerTestDefinition = cast(SlurmContainerTestDefinition, tr.test.test_definition)
        return [*cmd, *tdef.extra_srun_args, "--no-container-mount-home"]

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> list[str]:
        tdef: SlurmContainerTestDefinition = cast(SlurmContainerTestDefinition, tr.test.test_definition)
        srun_command_parts: list[str] = [*super().gen_nsys_command(tr), tdef.cmd_args.cmd]
        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return ["bash", "-c", f'"{" ".join(srun_command_parts)}"']
