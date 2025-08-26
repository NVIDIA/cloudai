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

import logging
from pathlib import Path
from typing import List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field

from cloudai.core import DockerImage, File, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


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


class OptimConfig(BaseModel):
    """Configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")
    use_precision_aware_optimizer: Optional[Union[bool, List[bool]]] = None


class Optim(BaseModel):
    """Optimizer configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")
    config: Optional[OptimConfig] = None


class Data(BaseModel):
    """Data configuration for NeMoRun."""

    model_config = ConfigDict(extra="allow")

    seq_length: Union[int, List[int]] = 8192
    micro_batch_size: Union[int, List[int]] = 1
    global_batch_size: Union[int, List[int]] = 1
    num_train_samples: Optional[int] = 1000


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
    val_check_interval: Union[int, float, list[Union[int, float]]] = 1000
    num_nodes: Optional[int] = None  # sweeps are done via TestRun.num_nodes
    devices: Optional[int] = None
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
    optim: Optim = Field(default_factory=Optim)


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

    def constraint_check(self, tr: TestRun) -> bool:
        """Check constraints for NeMoRun."""
        tp = cast(int, self.cmd_args.trainer.strategy.tensor_model_parallel_size)
        pp = cast(int, self.cmd_args.trainer.strategy.pipeline_model_parallel_size)
        cp = cast(int, self.cmd_args.trainer.strategy.context_parallel_size)
        vp = cast(Optional[int], self.cmd_args.trainer.strategy.virtual_pipeline_model_parallel_size)
        num_gpus = tr.nnodes * (self.cmd_args.trainer.devices if self.cmd_args.trainer.devices else 8)
        num_layers = cast(int, self.cmd_args.num_layers)
        dp = num_gpus // (tp * pp * cp)
        mbs = cast(int, self.cmd_args.data.micro_batch_size)
        gbs = cast(int, self.cmd_args.data.global_batch_size)

        constraint1 = num_gpus % (tp * pp * cp) == 0
        if not constraint1:
            logging.error(
                "Constraint 1 failed: num_gpus %% (tp * pp * cp) != 0. "
                f"Values: num_gpus={num_gpus}, tp={tp}, pp={pp}, cp={cp}"
            )

        constraint2 = True if vp is None else (num_layers // pp) % vp == 0
        if not constraint2:
            logging.error(
                "Constraint 2 failed: vp is not None and (num_layers // pp) %% vp != 0. "
                f"Values: num_layers={num_layers}, pp={pp}, vp={vp}"
            )

        constraint3 = dp != 0
        if not constraint3:
            logging.error(
                f"Constraint 3 failed: dp == 0. Values: dp={dp}, num_gpus={num_gpus}, tp={tp}, pp={pp}, cp={cp}"
            )

        constraint4 = gbs % (mbs * dp) == 0 if dp != 0 else False
        if not constraint4:
            logging.error(f"Constraint 4 failed: gbs %% (mbs * dp) != 0. Values: gbs={gbs}, mbs={mbs}, dp={dp}")

        return constraint1 and constraint2 and constraint3 and constraint4

    @property
    def update_num_train_samples(self) -> Optional[int]:
        """Calculate num_train_samples based on global_batch_size and max_steps."""
        gbs = self.cmd_args.data.global_batch_size
        max_steps = self.cmd_args.trainer.max_steps

        if isinstance(gbs, int) and isinstance(max_steps, int):
            return gbs * max_steps
        return None

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stderr_path = tr.output_path / "stderr.txt"
        if stderr_path.is_file():
            with stderr_path.open("r") as file:
                content = file.read()

                if "max_steps=" in content and "reached" in content:
                    return JobStatusResult(is_successful=True)

                missing_indicators = []
                if "max_steps=" not in content:
                    missing_indicators.append("'max_steps='")
                if "reached" not in content:
                    missing_indicators.append("'reached'")

                error_message = (
                    f"Missing success indicators in {stderr_path}: {', '.join(missing_indicators)}. "
                    "These keywords are expected to be present in stderr.txt when the NeMo training job "
                    "completes successfully. Please review the full stderr output. "
                    "Ensure that the NeMo training ran to completion and the logger output wasn't suppressed. "
                    "If the issue persists, contact the system administrator."
                )
                return JobStatusResult(is_successful=False, error_message=error_message)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stderr.txt file not found in the specified output directory {tr.output_path}. "
                "This file is expected to be created as part of the NeMo training job. "
                "Please ensure the job was submitted and executed properly. "
                f"You can try re-running the job manually and verify that {stderr_path} is created "
                "with the expected output. If the issue persists, contact the system administrator."
            ),
        )
