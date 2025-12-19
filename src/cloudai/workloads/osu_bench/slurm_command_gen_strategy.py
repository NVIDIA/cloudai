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

from .osu_bench import OSUBenchCmdArgs, OSUBenchTestDefinition

FULL_FLAG_UNSUPPORTED = [
    "osu_latency",
    "osu_latency_mt",
    "osu_latency_mp",
    "osu_bw",
    "osu_bibw",
    "osu_latency_persistent",
    "osu_bw_persistent",
    "osu_bibw_persistent",
    "osu_multi_lat",
    "osu_mbw_mr",
    "osu_put_latency",
    "osu_get_latency",
    "osu_acc_latency",
    "osu_get_acc_latency",
    "osu_cas_latency",
    "osu_fop_latency",
    "osu_put_bw",
    "osu_get_bw",
    "osu_put_bibw",
    "osu_init",
    "osu_hello",
]


class OSUBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for OSU Benchmark test on Slurm systems."""

    def _container_mounts(self) -> List[str]:
        return []

    def image_path(self) -> str:
        tdef: OSUBenchTestDefinition = cast(OSUBenchTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        args: OSUBenchCmdArgs = cast(OSUBenchCmdArgs, self.test_run.test.cmd_args)

        binary = f"{args.benchmarks_dir}/{args.benchmark}"
        srun_command_parts = [binary]

        for name, value in self.test_run.test.cmd_args_dict.items():
            if value is None:
                continue

            flag = f"--{name.replace('_', '-')}"

            argument = flag if isinstance(value, bool) and value else f"{flag} {value}"

            if name == "full" and args.benchmark in FULL_FLAG_UNSUPPORTED:
                continue

            srun_command_parts.append(argument)

        if self.test_run.test.extra_cmd_args:
            srun_command_parts.append(self.test_run.test.extra_args_str)

        return srun_command_parts
