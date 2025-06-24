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

from typing import Optional

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class NIXLBenchCmdArgs(CmdArgs):
    """Command line arguments for a NIXL Bench test."""

    docker_image_url: str
    etcd_endpoint: str
    path_to_benchmark: str


class NIXLBenchTestDefinition(TestDefinition):
    """Test definition for a NIXL Bench test."""

    cmd_args: NIXLBenchCmdArgs
    etcd_image_url: str
    _nixl_image: Optional[DockerImage] = None
    _etcd_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._nixl_image:
            self._nixl_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._nixl_image

    @property
    def etcd_image(self) -> DockerImage:
        if not self._etcd_image:
            self._etcd_image = DockerImage(url=self.etcd_image_url)
        return self._etcd_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos, self.etcd_image]
