# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from os.path import expandvars
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class MegatronRunCmdArgs(CmdArgs):
    """MegatronRun test command arguments."""

    docker_image_url: str = Field()
    run_script: Path = Field()

    global_batch_size: Optional[int] = 16
    hidden_size: Optional[int] = 4096
    max_position_embeddings: Optional[int] = 4096
    num_attention_heads: Optional[int] = 32
    num_layers: Optional[int] = 32
    pipeline_model_parallel_size: Optional[int] = 1
    recompute_activations: Optional[str] = None
    seq_length: Optional[int] = 4096
    tensor_model_parallel_size: Optional[int] = 2

    save: Optional[Path] = None
    load: Optional[Path] = None
    tokenizer_model: Optional[Path] = None

    @field_validator("save", "load", mode="before")
    @classmethod
    def expand_paths(cls, v: Optional[str]) -> Optional[Path]:
        if v is None:
            return None
        return Path(expandvars(v))

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
        args = self.model_dump(exclude_none=True, exclude={"docker_image_url", "run_script"})
        args = {f"--{k.replace('_', '-')}": v for k, v in args.items()}
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

    @model_validator(mode="after")
    def load_path_specified(self):
        to_check: list[Tuple[Path, str]] = []
        if self.cmd_args.load:
            to_check.append((self.cmd_args.load, "load"))
        if self.cmd_args.save:
            to_check.append((self.cmd_args.save, "save"))

        for path, field in to_check:
            self._check_path_is_mounted(path, field)

        return self

    def _check_path_is_mounted(self, path: Path, field: str) -> None:
        src_mount: Optional[str] = None
        for mount in self.extra_container_mounts:
            expanded = expandvars(mount)
            if ":" in expanded:
                src, dst = expanded.split(":")
                if dst == str(path.absolute()):
                    src_mount = src
                    break
            else:
                if expanded == str(path.absolute()):
                    src_mount = expanded
                    break

        if not src_mount:
            raise ValueError(
                f"Path {path} is not mounted in the container. Please check the 'extra_container_mounts' field."
            )

        if not Path(expandvars(src_mount)).exists():
            raise ValueError(f"Source path {src_mount} ({expandvars(src_mount)}) does not exist for {field}={path}.")
