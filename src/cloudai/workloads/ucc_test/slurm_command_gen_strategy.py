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

from .ucc import UCCCmdArgs, UCCTestDefinition


class UCCTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for UCC tests on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        return []

    def image_path(self, tr: TestRun) -> str | None:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, tr.test.test_definition)
        tdef_cmd_args: UCCCmdArgs = tdef.cmd_args

        srun_command_parts = ["/opt/hpcx/ucc/bin/ucc_perftest"]
        srun_command_parts.append(f"-c {tdef_cmd_args.collective}")
        srun_command_parts.append(f"-b {tdef_cmd_args.b}")
        srun_command_parts.append(f"-e {tdef_cmd_args.e}")
        srun_command_parts.append("-m cuda")
        srun_command_parts.append("-F")

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts
