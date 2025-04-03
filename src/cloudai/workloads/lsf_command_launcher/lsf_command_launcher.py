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

from typing import Optional, Union, List

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition

class LSFCmdLauncherCmdArgs(CmdArgs):
    """Command line arguments for a generic LSF Command Launcher test."""

    cmd: str
    hybrid_parallel_config: Optional[Union[str, List[str]]] = None
    chunk_size: Optional[Union[str, List[str]]] = None


class LSFCommandLauncherTestDefinition(TestDefinition):
    """Test definition for a generic LSF Command Launcher test."""

    cmd_args: LSFCmdLauncherCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)
