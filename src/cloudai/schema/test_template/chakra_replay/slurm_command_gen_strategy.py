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

from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        if tdef.cmd_args.trace_path:
            return [f"{tdef.cmd_args.trace_path}:{tdef.cmd_args.trace_path}"]
        return []

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _gen_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        if "container_name" not in slurm_args:
            slurm_args["container_name"] = "chakra_replay"

        srun_prefix = " ".join(self.gen_srun_prefix(slurm_args, tr))

        setup_command = (
            f"srun --ntasks-per-node=1 {srun_prefix} bash -c \"pip install pydot ; "
            "git clone https://github.com/TaekyungHeo/param.git -b theo/report ; "
            "cd param/et_replay ; "
            "pip install .\""
        )

        replay_command = (
            f"{srun_prefix} bash -c \"comm_replay --trace-type {cmd_args['trace_type']} "
            f"--trace-path {cmd_args['trace_path']} "
            f"--num-replays {cmd_args['num_replays']} "
            f"{tr.test.extra_cmd_args}\""
        )

        return f"{setup_command}\n{replay_command}"
