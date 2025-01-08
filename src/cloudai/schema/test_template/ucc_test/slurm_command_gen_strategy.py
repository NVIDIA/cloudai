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
from cloudai.test_definitions.ucc import UCCTestDefinition


class UCCTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for UCC tests on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: UCCTestDefinition = cast(UCCTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        srun_command_parts = ["/opt/hpcx/ucc/bin/ucc_perftest"]

        # Add collective, minimum bytes, and maximum bytes options if available
        if "collective" in cmd_args:
            srun_command_parts.append(f"-c {cmd_args['collective']}")
        if "b" in cmd_args:
            srun_command_parts.append(f"-b {cmd_args['b']}")
        if "e" in cmd_args:
            srun_command_parts.append(f"-e {cmd_args['e']}")

        # Append fixed string options for memory type and additional flags
        srun_command_parts.append("-m cuda")
        srun_command_parts.append("-F")

        # Append any extra command-line arguments provided
        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts
