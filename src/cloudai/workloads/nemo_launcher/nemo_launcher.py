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

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cloudai import DockerImage, GitRepo, Installable, PythonExecutable

from ...models.workload import CmdArgs, TestDefinition


class NumaMapping(BaseModel):
    """NUMA mapping configuration."""

    model_config = ConfigDict(extra="allow")
    enable: bool = True


class Cluster(BaseModel):
    """Cluster configuration."""

    model_config = ConfigDict(extra="allow")
    gpus_per_node: int = 8


class ExpManager(BaseModel):
    """Experiment manager configuration."""

    model_config = ConfigDict(extra="allow")
    create_checkpoint_callback: bool = False


class Trainer(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(extra="allow")
    max_steps: int = 20
    val_check_interval: int = 10
    log_every_n_steps: Literal["1", "2"] = "1"
    enable_checkpointing: bool = False


class TrainingModelData(BaseModel):
    """Training model data configuration."""

    model_config = ConfigDict(extra="allow")
    data_prefix: str = "[]"
    data_impl: str = "mock"


class TrainingModel(BaseModel):
    """Training model configuration."""

    model_config = ConfigDict(extra="allow")
    global_batch_size: int = 128
    micro_batch_size: int = 2
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 4
    data: TrainingModelData = Field(default_factory=TrainingModelData)


class TrainingRun(BaseModel):
    """Training run configuration."""

    model_config = ConfigDict(extra="allow")
    time_limit: str = "3:00:00"
    name: str = "run"


class Training(BaseModel):
    """Training configuration."""

    model_config = ConfigDict(extra="allow")
    values: str = "gpt3/40b_improved"
    exp_manager: ExpManager = Field(default_factory=ExpManager)
    trainer: Trainer = Field(default_factory=Trainer)
    model: TrainingModel = Field(default_factory=TrainingModel)
    run: TrainingRun = Field(default_factory=TrainingRun)


class NeMoLauncherCmdArgs(CmdArgs):
    """NeMoLauncher test command arguments."""

    launcher_script: str = "launcher_scripts/main.py"
    docker_image_url: str = "nvcr.io/nvidia/nemo:24.12.01"
    stages: str = '["training"]'
    numa_mapping: NumaMapping = Field(default_factory=NumaMapping)
    cluster: Cluster = Field(default_factory=Cluster)
    training: Training = Field(default_factory=Training)


class NeMoLauncherTestDefinition(TestDefinition):
    """Test object for NeMoLauncher."""

    cmd_args: NeMoLauncherCmdArgs
    launcher_repo: GitRepo = GitRepo(
        url="https://github.com/NVIDIA/NeMo-Framework-Launcher.git", commit="599ecfcbbd64fd2de02f2cc093b1610d73854022"
    )
    _docker_image: Optional[DockerImage] = None
    _python_executable: Optional[PythonExecutable] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def python_executable(self) -> PythonExecutable:
        if not self._python_executable:
            self._python_executable = PythonExecutable(
                GitRepo(url=self.launcher_repo.url, commit=self.launcher_repo.commit)
            )
        return self._python_executable

    @property
    def installables(self) -> list[Installable]:
        """Get list of installable objects."""
        return [self.docker_image, self.python_executable]
