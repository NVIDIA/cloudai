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

from typing import Literal, Optional

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition


class NimCmdArgs(CmdArgs):
    """NIM test command arguments."""

    docker_image_url: str = "nvcr.io/nvidia/tritonserver:25.01-py3-sdk"
    served_model_name: str
    endpoint_type: Literal["chat"] = "chat"
    service_kind: Literal["openai"] = "openai"
    streaming: bool = True
    leader_ip: str
    port: int = 8000
    num_prompts: int = 20
    input_sequence_length: int = 128
    output_sequence_length: int = 128
    concurrency: int = 1
    tokenizer: str


class NimTestDefinition(TestDefinition):
    """Test object for Nim."""

    cmd_args: NimCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
