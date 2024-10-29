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


class NeMoRunCmdArgs(CmdArgs):
    """NeMoRun test command arguments."""

    docker_image_url: str = "nvcr.io/nvidia/nemo:24.09"
    task: Optional[str] = None
    recipe_name: Optional[str] = None


class NeMoRunTestDefinition(TestDefinition):
    """Test object for NeMoLauncher."""

    cmd_args: NeMoRunCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        """Get list of installable objects."""
        return [self.docker_image]
