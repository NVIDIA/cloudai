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

from typing import Any, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.slurm_container import SlurmContainerTestDefinition


class SlurmContainerCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for generic Slurm container tests."""

    def gen_srun_prefix(self, slurm_args: dict[str, Any], tr: TestRun) -> list[str]:
        tdef: SlurmContainerTestDefinition = cast(SlurmContainerTestDefinition, tr.test.test_definition)
        slurm_args["image_path"] = tdef.docker_image.installed_path
        mounts = tdef.container_mounts(self.system.install_path)
        mounts.append(f"{tr.output_path.absolute()}:/cloudai_run_results")
        slurm_args["container_mounts"] = ",".join(mounts)

        cmd = super().gen_srun_prefix(slurm_args, tr)
        return cmd + ["--no-container-mount-home"]

    def generate_test_command(self, env_vars: dict[str, str], cmd_args: dict[str, str], tr: TestRun) -> list[str]:
        srun_command_parts: list[str] = []
        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts
