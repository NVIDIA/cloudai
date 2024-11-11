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

from cloudai import CmdArgs, Installable, TestDefinition
from cloudai.installer.installables import DockerImage, GitRepo


class SlurmContainerCmdArgs(CmdArgs):
    """Command line arguments for a generic Slurm container test."""

    docker_image_url: str
    repository_url: str
    repository_commit_hash: str
    mcore_vfm_repo: str
    mcore_vfm_commit_hash: str


class SlurmContainerTestDefinition(TestDefinition):
    """Test definition for a generic Slurm container test."""

    cmd_args: SlurmContainerCmdArgs

    _docker_image: Optional[DockerImage] = None
    _git_repo: Optional[GitRepo] = None
    _mcore_git_repo: Optional[GitRepo] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def git_repo(self) -> GitRepo:
        if not self._git_repo:
            self._git_repo = GitRepo(
                git_url=self.cmd_args.repository_url, commit_hash=self.cmd_args.repository_commit_hash
            )

        return self._git_repo

    @property
    def mcore_vfm_git_repo(self) -> GitRepo:
        if not self._mcore_git_repo:
            self._mcore_git_repo = GitRepo(
                git_url=self.cmd_args.mcore_vfm_repo, commit_hash=self.cmd_args.mcore_vfm_commit_hash
            )

        return self._mcore_git_repo

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.git_repo, self.mcore_vfm_git_repo]

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)
