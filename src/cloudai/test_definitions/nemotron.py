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

from pydantic import Field

from cloudai import Installable
from cloudai.installer.installables import DockerImage

from .grok import GrokCmdArgs, GrokTestDefinition
from .jax_toolbox import JaxFdl, SetupFlags, XLAFlags


class NemotronFdl(JaxFdl):
    """NemotronFdl."""

    num_layers: int = 2
    checkpoint_policy: str = '"save_dot_except_logits_ffn1"'
    num_gpus: int = 8
    percore_batch_size: float = 0.25  # type: ignore
    use_repeated_layer: bool = True


class NemotrolXLAFlags(XLAFlags):
    """NemotrolXLAFlags."""

    xla_gpu_all_gather_combine_threshold_bytes: int = 3221225472
    xla_gpu_all_reduce_combine_threshold_bytes: int = 3221225472
    xla_gpu_reduce_scatter_combine_threshold_bytes: int = 6291456


class NemotronSetupFlags(SetupFlags):
    """NemotronSetupFlags."""

    vocab_path: str = "/mnt/vocab"


class NemotronCmdArgs(GrokCmdArgs):
    """NemotronCmdArgs."""

    xla_flags: NemotrolXLAFlags = Field(default_factory=NemotrolXLAFlags)
    setup_flags: NemotronSetupFlags = Field(default_factory=NemotronSetupFlags)  # type: ignore


class NemotronTestDefinition(GrokTestDefinition):
    """NemotronTestDefinition."""

    cmd_args: NemotronCmdArgs  # type: ignore
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
