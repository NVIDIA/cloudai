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

from pydantic import BaseModel, Field

from cloudai import CmdArgs, TestDefinition
from cloudai.installer.installables import DockerImage, Installable


class Trainer(BaseModel):
    """Trainer configuration for NeMoRun."""

    max_steps: int = 1168251
    val_check_interval: int = 1000
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    context_parallelism: int = 2
    sequence_parallelism: bool = False
    num_nodes: int = 1
    num_gpus_per_node: int = 8


class LogCkpt(BaseModel):
    """Logging checkpoint configuration for NeMoRun."""

    save_on_train_epoch_end: bool = Field(default=False)
    save_last: bool = Field(default=False)


class Log(BaseModel):
    """Base logging configuration for NeMoRun."""

    ckpt: LogCkpt = Field(default_factory=LogCkpt)


class NeMoRunCmdArgs(CmdArgs):
    """NeMoRun test command arguments."""

    docker_image_url: str
    task: str
    recipe_name: str
    trainer: Trainer = Field(default_factory=Trainer)
    log: Log = Field(default_factory=Log)


class NeMoRunTestDefinition(TestDefinition):
    """NeMoRun test definition."""

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
