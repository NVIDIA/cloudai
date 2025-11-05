# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ddlb import DDLBTestDefinition


class DDLBTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for DDLB tests on Slurm systems."""

    def _container_mounts(self) -> List[str]:
        return []

    def image_path(self) -> str | None:
        tdef: DDLBTestDefinition = cast(DDLBTestDefinition, self.test_run.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        tdef: DDLBTestDefinition = cast(DDLBTestDefinition, self.test_run.test.test_definition)
        srun_command_parts = ["python ddlb/cli/benchmark.py"]
        ddlb_test_args = tdef.cmd_args.model_dump().keys()
        for arg in ddlb_test_args:
            if arg == "docker_image_url":
                continue

            value = getattr(tdef.cmd_args, arg)
            if value is None:
                continue

            match arg:
                case "m" | "n" | "k":
                    srun_command_parts.append(f"-{arg} {value}")
                case "num_iterations" | "num_warmups":
                    srun_command_parts.append(f"--{arg.replace('_', '-')} {value}")
                case _:
                    srun_command_parts.append(f"--{arg} {value}")

        if self.test_run.test.extra_cmd_args:
            srun_command_parts.append(self.test_run.test.extra_cmd_args)

        return srun_command_parts

    def gen_srun_success_check(self) -> str:
        output_file = self.test_run.output_path / "stdout.txt"
        return f'grep -q "Benchmark Results" {output_file} && echo 1 || echo 0'
