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
    """Plugin configuration."""

    model_config = ConfigDict(extra="allow")
    fp8: Optional[str] = None
    fp8_margin: Optional[int] = None
    fp8_amax_history_len: Optional[int] = None
    fp8_amax_compute_algo: Optional[str] = None
    fp8_wgrad: Optional[bool] = None
    fp8_params: Optional[bool] = None
    grad_reduce_in_fp32: Optional[bool] = None


class Data(BaseModel):
    """Data configuration."""

    model_config = ConfigDict(extra="allow")
    micro_batch_size: Union[int, List[int]] = 1
    global_batch_size: Union[int, List[int]] = 1


class TrainerStrategy(BaseModel):
    """Trainer strategy configuration."""

    model_config = ConfigDict(extra="allow")
    tensor_model_parallel_size: Union[int, List[int]] = 1
    pipeline_model_parallel_size: Union[int, List[int]] = 1
    context_parallel_size: Union[int, List[int]] = 2
    virtual_pipeline_model_parallel_size: Optional[Union[int, List[int]]] = None


class TrainerCallbacksCommOverlap(BaseModel):
    """Comm overlap callback configuration."""

    model_config = ConfigDict(extra="allow")


class TrainerCallbacks(BaseModel):
    """Trainer callbacks configuration."""

    model_config = ConfigDict(extra="allow")
    comm_overlap: Optional[TrainerCallbacksCommOverlap] = None
    # Add other callbacks as needed


class Trainer(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(extra="allow")
    max_steps: Union[int, List[int]] = 100
    val_check_interval: Union[int, List[int]] = 1000
    num_nodes: Optional[Union[int, List[int]]] = None
    strategy: TrainerStrategy = Field(default_factory=TrainerStrategy)
    plugins: Optional[Plugin] = None
    callbacks: Optional[Union[str, list[str]]] = None


class LogCkpt(BaseModel):
    """Checkpoint logging configuration."""

    model_config = ConfigDict(extra="allow")
    save_on_train_epoch_end: Optional[bool] = None
    save_last: Optional[bool] = None


class LogTensorboard(BaseModel):
    """Tensorboard logging configuration."""

    model_config = ConfigDict(extra="allow")
    save_dir: Union[str, Path] = Field(default="logs")
    name: Optional[str] = Field(default="default")
    log_dir: Optional[Union[str, Path]] = None


class Log(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="allow")
    ckpt: Optional[LogCkpt] = None
    tensorboard: Optional[LogTensorboard] = None


class NeMoRunBaseCmdArgs(CmdArgs):
    """Common NeMoRun command arguments."""

    docker_image_url: str
    task: str
    recipe_name: str
    num_layers: Optional[int] = None
    trainer: Trainer = Field(default_factory=Trainer)
    log: Log = Field(default_factory=Log)
    data: Data = Field(default_factory=Data)


class NeMoRunLLAMA3PretrainCmdArgs(NeMoRunBaseCmdArgs):
    """LLAMA3.1 pretraining command arguments."""

    # LLAMA3.1 experiment parameters
    cluster: str = Field(..., description="Cluster identifier")
    gsw_version: str = Field(..., description="GSW version for the job")
    compute_dtype: str = Field(..., description="Compute precision type (e.g. 'fp8', 'bf16')")
    model_size: str = Field(..., description="Model size (e.g. '8b', '70b', '405b')")
    num_gpus: int = Field(..., description="Total number of GPUs available")
    gpus_per_node: int = Field(..., description="Number of GPUs per node")

    # Slurm executor / container related settings
    account: str = Field(..., description="Slurm account")
    partition: str = Field(..., description="Slurm partition")
    log_dir: Union[str, Path] = Field("logs", description="Directory for logs")
    time_limit: str = Field(..., description="Job time limit (e.g. '00:15:00')")
    container_image: str = Field(..., description="Docker container image URL")

    # Environment and mount settings
    custom_mounts: Optional[List[str]] = Field(default_factory=list, description="Custom mounts for the job")
    custom_env_vars: Optional[dict] = Field(default_factory=dict, description="Custom environment variables")

    # Authentication and home directory settings
    hf_token: str = Field(..., description="Hugging Face authentication token")
    nemo_home: str = Field(..., description="Path to the Nemo home directory")

    # Experiment tracking settings
    wandb_key: str = Field(..., description="Weights & Biases API key")
    tensorboard: Optional[Union[str, Path]] = Field("logs", description="Tensorboard logging directory")
    wandb: bool = Field(False, description="Enable Weights & Biases logging")
    wandb_prj_name: Optional[str] = Field(None, description="W&B project name")
    wandb_job_name: Optional[str] = Field(None, description="W&B job name")

    # Additional run options
    enable_profiling: bool = Field(False, description="Enable profiling")
    enable_nccltrace: bool = Field(False, description="Enable NCCL trace logging")
    disable_perfrun: bool = Field(False, description="Disable performance run")
    dryrun: bool = Field(False, description="Perform a dry run without executing the experiment")
    gpu: str = Field(..., description="GPU type identifier (e.g. 'h100', 'b200')")
    optimization_name: Optional[str] = Field(None, description="Name of the optimization method")
    optimization_code: Optional[str] = Field(None, description="Optimization code snippet to execute")


class NeMoRunTestDefinition(TestDefinition):
    """NeMoRun test definition."""

    cmd_args: NeMoRunBaseCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "nemo_run/cloudai_nemorun.py")
    llmb_script: File = File(Path(__file__).parent.parent / "nemo_run/llmb_llama3.py")

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.container_image)
        return self._docker_image

    @property
    def installables(self) -> List[Installable]:
        return [self.docker_image, self.script, self.llmb_script]

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
        constraint3 = gbs % (mbs * dp) == 0

        return constraint1 and constraint2 and constraint3
