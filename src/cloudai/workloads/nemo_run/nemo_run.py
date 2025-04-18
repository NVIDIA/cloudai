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

from pathlib import Path
from typing import List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field

from cloudai import CmdArgs, DockerImage, File, Installable, TestDefinition


class Plugin(BaseModel):
    """Plugin configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    fp8: Optional[str] = None
    fp8_margin: Optional[int] = None
    fp8_amax_history_len: Optional[int] = None
    fp8_amax_compute_algo: Optional[str] = None
    fp8_wgrad: Optional[bool] = None
    fp8_params: Optional[bool] = None
    grad_reduce_in_fp32: Optional[bool] = None


class Data(BaseModel):
    """Data configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    seq_length: Union[int, List[int]] = 8192
    micro_batch_size: Union[int, List[int]] = 1
    global_batch_size: Union[int, List[int]] = 1


class TrainerStrategy(BaseModel):
    """Trainer strategy configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    tensor_model_parallel_size: Union[int, List[int]] = 1
    pipeline_model_parallel_size: Union[int, List[int]] = 1
    context_parallel_size: Union[int, List[int]] = 2
    virtual_pipeline_model_parallel_size: Optional[Union[int, List[int]]] = None


class Trainer(BaseModel):
    """Trainer configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    max_steps: Union[int, List[int]] = 100
    val_check_interval: Union[int, List[int]] = 1000
    num_nodes: Optional[Union[int, List[int]]] = None
    strategy: TrainerStrategy = Field(default_factory=TrainerStrategy)
    plugins: Optional[Plugin] = None
    callbacks: Optional[Union[str, list[str]]] = None


class LogCkpt(BaseModel):
    """Logging checkpoint configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    save_on_train_epoch_end: Optional[bool] = Field(default=None)
    save_last: Optional[bool] = Field(default=None)


class LogTensorboard(BaseModel):
    """Logging tensorboard configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")
    save_dir: Union[str, Path] = Field(default="logs")
    name: Optional[str] = Field(default="default")


class Log(BaseModel):
    """Base logging configuration for NeMoRun."""

    ckpt: Optional[LogCkpt] = Field(default=None)
    tensorboard: Optional[LogTensorboard] = Field(default=None)

    model_config = ConfigDict(extra="allow")


class NeMoRunCmdArgs(CmdArgs):
    """NeMoRun test command arguments."""

    docker_image_url: str
    task: str
    recipe_name: str
    num_layers: Optional[int] = None
    trainer: Trainer = Field(default_factory=Trainer)
    log: Log = Field(default_factory=Log)
    data: Data = Field(default_factory=Data)


class NeMoRunTestDefinition(TestDefinition):
    """NeMoRun test definition."""

    cmd_args: NeMoRunCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "nemo_run/cloudai_nemorun.py")

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        """Get list of installable objects."""
        return [self.docker_image, self.script]

    @property
    def constraint_check(self) -> bool:
        """Check constraints for NeMoRun."""
        tp = cast(int, self.cmd_args.trainer.strategy.tensor_model_parallel_size)
        pp = cast(int, self.cmd_args.trainer.strategy.pipeline_model_parallel_size)
        cp = cast(int, self.cmd_args.trainer.strategy.context_parallel_size)
        vp = cast(Optional[int], self.cmd_args.trainer.strategy.virtual_pipeline_model_parallel_size)
        num_nodes = cast(int, self.cmd_args.trainer.num_nodes)
        num_gpus = num_nodes * 8
        num_layers = cast(int, self.cmd_args.num_layers)
        dp = num_gpus // (tp * pp * cp)
        mbs = cast(int, self.cmd_args.data.micro_batch_size)
        gbs = cast(int, self.cmd_args.data.global_batch_size)

        constraint1 = num_gpus % (tp * pp * cp) == 0
        constraint2 = True if vp is None else (num_layers // pp) % vp == 0
        constraint3 = dp != 0
        constraint4 = gbs % (mbs * dp) == 0 if dp != 0 else False

        return constraint1 and constraint2 and constraint3 and constraint4
