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

from typing import Any, Dict, List

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, str], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        image_path = cmd_args["docker_image_url"]
        container_mounts = f"{cmd_args['trace_path']}:{cmd_args['trace_path']}"

        base_args.update({"image_path": image_path, "container_mounts": container_mounts})

        return base_args

    def generate_test_command(self, env_vars: Dict[str, str], cmd_args: Dict[str, str], tr: TestRun) -> List[str]:
        srun_command_parts = [
            "python /workspace/param/train/comms/pt/commsTraceReplay.py",
            f'--trace-type {cmd_args["trace_type"]}',
            f'--trace-path {cmd_args["trace_path"]}',
            f'--backend {cmd_args["backend"]}',
            f'--device {cmd_args["device"]}',
            tr.test.extra_cmd_args,
        ]
        return srun_command_parts
