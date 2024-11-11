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

from typing import Optional

from cloudai import CmdArgs, TestDefinition
from cloudai.installer.installables import DockerImage, Installable


class ChakraReplayCmdArgs(CmdArgs):
    """ChakraReplay test command arguments."""

    docker_image_url: str
    mpi: str = "pmix"
    trace_type: str = "et"
    trace_path: Optional[str] = None
    num_replays: int = 1


class ChakraReplayTestDefinition(TestDefinition):
    """Test object for ChakraReplay."""

    cmd_args: ChakraReplayCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
