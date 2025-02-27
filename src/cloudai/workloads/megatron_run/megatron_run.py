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

from pathlib import Path
from typing import Optional, Union

from pydantic import Field, model_validator

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition


class MegatronRunCmdArgs(CmdArgs):
    """MegatronRun test command arguments."""

    docker_image_url: str = Field(exclude=True)
    run_script: Path = Field(exclude=True)

    global_batch_size: Optional[int] = 16
    hidden_size: Optional[int] = 4096
    max_position_embeddings: Optional[int] = 4096
    num_attention_heads: Optional[int] = 32
    num_layers: Optional[int] = 32
    pipeline_model_parallel_size: Optional[int] = 1
    recompute_activations: Optional[str] = ""
    seq_length: Optional[int] = 4096
    tensor_model_parallel_size: Optional[int] = 2

    save: Optional[Path] = None
    load: Optional[Path] = None
    tokenizer_model: Optional[Path] = None

    @model_validator(mode="after")
    def no_dashed_args(self):
        if not self.model_extra:
            return self

        dashed_args = {k for k in self.model_extra if "-" in k}
        if dashed_args:
            raise ValueError(f"Dashed arguments found: {dashed_args}. Replace with underscores.")

        return self

    @property
    def cmd_args(self) -> dict[str, Union[str, list[str]]]:
        args = self.model_dump(exclude_none=True)
        args = {f'--{k.replace("_", "-")}': v for k, v in args.items()}
        return args


class MegatronRunTestDefinition(TestDefinition):
    """MegatronRun test definition."""

    cmd_args: MegatronRunCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def cmd_args_dict(self) -> dict[str, Union[str, list[str]]]:
        return self.cmd_args.cmd_args

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]
