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

from typing import Literal, Optional, Union

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class UCCCmdArgs(CmdArgs):
    """UCC test command arguments."""

    docker_image_url: str
    collective: Union[
        Literal[
            "allgather",
            "allgatherv",
            "allreduce",
            "alltoall",
            "alltoallv",
            "barrier",
            "bcast",
            "gather",
            "gatherv",
            "reduce",
            "reduce_scatter",
            "reduce_scatterv",
            "scatter",
            "scatterv",
            "memcpy",
            "reducedt",
            "reducedt_strided",
        ],
        list[
            Literal[
                "allgather",
                "allgatherv",
                "allreduce",
                "alltoall",
                "alltoallv",
                "barrier",
                "bcast",
                "gather",
                "gatherv",
                "reduce",
                "reduce_scatter",
                "reduce_scatterv",
                "scatter",
                "scatterv",
                "memcpy",
                "reducedt",
                "reducedt_strided",
            ]
        ],
    ] = "alltoall"
    b: Union[int, list[int]] = 1
    e: Union[str, list[str]] = "8M"


class UCCTestDefinition(TestDefinition):
    """Test object for UCC."""

    cmd_args: UCCCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
