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

from .osu_bench import OSUBenchTestDefinition, OSUBenchCmdArgs


class OSUBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    def _container_mounts(self) -> List[str]:
        return []

    def image_path(self) -> str | None:
        tdef: OSUBenchTestDefinition = cast(OSUBenchTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        args: OSUBenchCmdArgs = cast(OSUBenchCmdArgs, self.test_run.test.cmd_args)

        binary = f'{args.location}/{args.benchmark}'
        cmd = [binary]

        for name, value in args.get_args().items():
            cmd.append(f"{name} {value}")

        # Always print full format listing of results.
        cmd.append('-f')

        return cmd
