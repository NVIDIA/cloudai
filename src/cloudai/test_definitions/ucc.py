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

from typing import Literal

from cloudai import CmdArgs, TestDefinition
from cloudai._core.test import DockerImage, Installable


class UCCCmdArgs(CmdArgs):
    """UCC test command arguments."""

    docker_image_url: str = "nvcr.io/nvidia/pytorch:24.02-py3"
    collective: Literal[
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
    ] = "alltoall"
    b: int = 1
    e: str = "8M"


class UCCTestDefinition(TestDefinition):
    """Test object for UCC."""

    cmd_args: UCCCmdArgs

    @property
    def docker_image(self) -> DockerImage:
        return DockerImage(url=self.cmd_args.docker_image_url)

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
