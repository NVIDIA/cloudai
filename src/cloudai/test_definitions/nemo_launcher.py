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

from pydantic import BaseModel, ConfigDict, Field

from cloudai import CmdArgs, TestDefinition


class NumaMapping(BaseModel):
    """NUMA mapping configuration."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = True


class Cluster(BaseModel):
    """Cluster configuration."""

    model_config = ConfigDict(extra="forbid")
    gpus_per_node: int = 8


class ExpManager(BaseModel):
    """Experiment manager configuration."""

    model_config = ConfigDict(extra="forbid")
    create_checkpoint_callback: bool = False


class Trainer(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(extra="forbid")
    max_steps: int = 400
    val_check_interval: int = 100
    log_every_n_steps: Literal["1", "2"] = "1"
    enable_checkpointing: bool = False


class TrainingModelData(BaseModel):
    """Training model data configuration."""

    model_config = ConfigDict(extra="forbid")
    data_prefix: str = "[]"
    data_impl: str = "mock"


class TrainingModel(BaseModel):
    """Training model configuration."""

    model_config = ConfigDict(extra="forbid")
    global_batch_size: int = 128
    micro_batch_size: int = 2
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 4
    data: TrainingModelData = Field(default_factory=TrainingModelData)


class TrainingRun(BaseModel):
    """Training run configuration."""

    model_config = ConfigDict(extra="forbid")
    time_limit: str = "3:00:00"
    name: str = "run"


class Training(BaseModel):
    """Training configuration."""

    model_config = ConfigDict(extra="forbid")
    values: str = "gpt3/40b_improved"
    exp_manager: ExpManager = Field(default_factory=ExpManager)
    trainer: Trainer = Field(default_factory=Trainer)
    model: TrainingModel = Field(default_factory=TrainingModel)
    run: TrainingRun = Field(default_factory=TrainingRun)


class NeMoLauncherCmdArgs(CmdArgs):
    """NeMoLauncher test command arguments."""

    repository_url: str = "https://github.com/NVIDIA/NeMo-Framework-Launcher.git"
    repository_commit_hash: str = "cf411a9ede3b466677df8ee672bcc6c396e71e1a"
    docker_image_url: str = "nvcr.io/nvidia/nemo:24.01.01"
    stages: str = '["training"]'
    data_dir: str = "~"
    numa_mapping: NumaMapping = Field(default_factory=NumaMapping)
    cluster: Cluster = Field(default_factory=Cluster)
    training: Training = Field(default_factory=Training)


class NeMoLauncherTestDefinition(TestDefinition):
    """Test object for NeMoLauncher."""

    cmd_args: NeMoLauncherCmdArgs
