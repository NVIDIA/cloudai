# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .vllm import VllmCmdArgs, VllmTestDefinition


class VllmSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for vLLM on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        return []

    def generate_test_command(self) -> List[str]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        tdef_cmd_args: VllmCmdArgs = tdef.cmd_args
        # TODO: Implement full command generation with bash script
        return [f"vllm serve {tdef_cmd_args.model}"]

    def get_vllm_serve_command(self) -> list[str]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        tdef_cmd_args: VllmCmdArgs = tdef.cmd_args
        return ["vllm", "serve", tdef_cmd_args.model, "--port", str(tdef_cmd_args.port)]
