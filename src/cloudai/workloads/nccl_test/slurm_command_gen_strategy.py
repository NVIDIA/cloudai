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
from typing import Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.nccl_test import NCCLTestDefinition


class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NCCL tests on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        return []

    def image_path(self, tr: TestRun) -> str | None:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        srun_command_parts = [f"{tdef.cmd_args.subtest_name}"]
        nccl_test_args = tdef.cmd_args.model_dump().keys()
        for arg in nccl_test_args:
            if arg in {"docker_image_url", "subtest_name"}:
                continue

            if len(arg) > 1:
                srun_command_parts.append(f"--{arg} {getattr(tdef.cmd_args, arg)}")
            else:
                srun_command_parts.append(f"-{arg} {getattr(tdef.cmd_args, arg)}")

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts

    def gen_srun_success_check(self, tr: TestRun) -> str:
        output_file = Path(tr.output_path) / "stdout.txt"
        return f'grep -q "Avg bus bandwidth" {output_file} && echo 1 || echo 0'
