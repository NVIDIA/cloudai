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

from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, self.test_run.test)
        if tdef.cmd_args.trace_path:
            return [f"{tdef.cmd_args.trace_path}:{tdef.cmd_args.trace_path}"]
        return []

    def image_path(self) -> str | None:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, self.test_run.test)

        additional_cmd_args_dict = tdef.cmd_args.model_extra or {}

        srun_command_parts: list[str] = [
            "comm_replay",
            f"--trace-type {tdef.cmd_args.trace_type}",
            f"--trace-path {tdef.cmd_args.trace_path}",
            f"--num-replays {tdef.cmd_args.num_replays}",
            *[f"--{k.replace('_', '-')} {v}" for k, v in additional_cmd_args_dict.items()],
            self.test_run.test.extra_args_str,
        ]
        return srun_command_parts
