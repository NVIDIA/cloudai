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

from .jax_toolbox import JaxFdl, JaxToolboxCmdArgs, JaxToolboxTestDefinition, PreTest, SetupFlags, XLAFlags


class GPTFdl(JaxFdl):
    """GPT FDL configuration."""

    num_groups: int = 64


class GPTXLAFlags(XLAFlags):
    """GPT XLA flags."""

    xla_gpu_all_reduce_combine_threshold_bytes: int = 447741952
    xla_gpu_enable_while_loop_double_buffering: bool = False


class GPTSetupFlags(SetupFlags):
    """GPT setup flags."""

    gpt_vocab_path: str = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"


class GPTCmdArgs(JaxToolboxCmdArgs):
    """GPT JAX Toolbox test command arguments."""

    fdl_config: str
    fdl: GPTFdl = Field(default_factory=GPTFdl)
    pre_test: PreTest = Field(default_factory=PreTest)
    xla_flags: GPTXLAFlags = Field(default_factory=GPTXLAFlags)
    setup_flags: GPTSetupFlags = Field(default_factory=GPTSetupFlags)


class GPTTestDefinition(JaxToolboxTestDefinition):
    """Test object for GPT."""

    cmd_args: GPTCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def cmd_args_dict(self):
        d = self.cmd_args.model_dump()
        res = {}
        for k, v in d.items():
            if k in {"pre_test", "docker_image_url", "load_container", "output_path"}:
                res[k] = v
            else:
                if k == "xla_flags":
                    k = k.upper()
                res[f"GPT.{k}"] = v
        return res

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
