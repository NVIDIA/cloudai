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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        if tdef.cmd_args.trace_path:
            return [f"{tdef.cmd_args.trace_path}:{tdef.cmd_args.trace_path}"]
        return []

    def image_path(self, tr: TestRun) -> str | None:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)

        additional_cmd_args_dict = tdef.cmd_args.model_extra or {}

        srun_command_parts = [
            "comm_replay",
            f"--trace-type {tdef.cmd_args.trace_type}",
            f"--trace-path {tdef.cmd_args.trace_path}",
            f"--num-replays {tdef.cmd_args.num_replays}",
            *[f"--{k.replace('_', '-')} {v}" for k, v in additional_cmd_args_dict.items()],
            tr.test.extra_cmd_args,
        ]
        return srun_command_parts
