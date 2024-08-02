#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CmdArgs(BaseModel):
    """Test command arguments."""

    model_config = ConfigDict(extra="forbid")


class TestDefinition(BaseModel):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    extra_env_vars: dict[str, str] = {}
    extra_cmd_args: str = ""


class UCCCmdArgs(CmdArgs):
    """UCC test command arguments."""

    docker_image_url: str = Field(default="nvcr.io/nvidia/pytorch:24.02-py3")
    collective: Literal[
        "allgather",
        "allgatherv",
        "allreduce",
        "alltoall",
        "alltoallv",
        "barrier",
        "bcast",
        "gather",
        "gatherv",
        "reduce",
        "reduce_scatter",
        "reduce_scatterv",
        "scatter",
        "scatterv",
        "memcpy",
        "reducedt",
        "reducedt_strided",
    ] = Field(default="alltoall")
    b: int = Field(default=1)
    e: str = Field(default="8M")


class UCCTestDefinition(TestDefinition):
    """Test object for UCC."""

    cmd_args: UCCCmdArgs


class NCCLCmdArgs(CmdArgs):
    """NCCL test command arguments."""

    docker_image_url: str = Field(default="nvcr.io/nvidia/pytorch:24.02-py3")
    subtest_name: Literal[
        "all_reduce_perf_mpi",
        "all_gather_perf_mpi",
        "alltoall_perf_mpi",
        "broadcast_perf_mpi",
        "gather_perf_mpi",
        "hypercube_perf_mpi",
        "reduce_perf_mpi",
        "reduce_scatter_perf_mpi",
        "scatter_perf_mpi",
        "sendrecv_perf_mpi",
    ] = Field(default="all_reduce_perf_mpi")
    nthreads: int = Field(default=1)
    ngpus: int = Field(default=1)
    minbytes: str = Field(default="32M")
    maxbytes: str = Field(default="32M")
    stepbytes: str = Field(default="1M")
    op: Literal["sum", "prod", "min", "max", "avg", "all"] = Field(default="sum")
    datatype: str = Field(default="float")
    root: int = Field(default=0)
    iters: int = Field(default=20)
    warmup_iters: int = Field(default=5)
    agg_iters: int = Field(default=1)
    average: int = Field(default=1)
    parallel_init: int = Field(default=0)
    check: int = Field(default=1)
    blocking: int = Field(default=0)
    cudagraph: int = Field(default=0)


class NCCLTestDefinition(TestDefinition):
    """Test object for NCCL."""

    cmd_args: NCCLCmdArgs


class ChakraReplayCmdArgs(CmdArgs):
    """ChakraReplay test command arguments."""

    docker_image_url: str = Field(default="DOCKER_IMAGE_URL")
    mpi: str = Field(default="pmix")
    trace_type: str = Field(default="et")
    trace_path: Optional[str] = None
    backend: str = Field(default="nccl")
    device: str = Field(default="cuda")


class ChakraReplayTestDefinition(TestDefinition):
    """Test object for ChakraReplay."""

    cmd_args: ChakraReplayCmdArgs


class SleepCmdArgs(CmdArgs):
    """Sleep test command arguments."""

    seconds: int = Field(default=5)


class SleepTestDefinition(TestDefinition):
    """Test object for Sleep."""

    cmd_args: SleepCmdArgs


class NumaMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = Field(default=True)


class Cluster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpus_per_node: int = Field(default=8)

    @field_validator("gpus_per_node")
    def validate_gpus_per_node(cls, value):
        valid_values = {4, 8, 16}
        if value not in valid_values:
            raise ValueError(f"gpus_per_node must be one of {valid_values}")
        return value


class ExpManager(BaseModel):
    model_config = ConfigDict(extra="forbid")
    create_checkpoint_callback: bool = Field(default=False)


class Trainer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = 400
    val_check_interval: int = 100
    log_every_n_steps: Literal["1", "2"] = Field(default="1")
    enable_checkpointing: bool = Field(default=False)

    @field_validator("val_check_interval")
    def validate_val_check_interval(cls, value):
        valid_values = {100, 500, 1000, 2000}
        if value not in valid_values:
            raise ValueError(f"val_check_interval must be one of {valid_values}")
        return value


class TrainingModelData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_prefix: Literal["[\"1.0\",'${data_dir}/my-gpt3_00_text_document']"] = Field(
        default="[\"1.0\",'${data_dir}/my-gpt3_00_text_document']"
    )


class TrainingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_batch_size: int = 128
    micro_batch_size: int = 2
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 4
    data: TrainingModelData = Field(default_factory=TrainingModelData)

    @field_validator("micro_batch_size")
    def validate_micro_batch_size(cls, value):
        valid_values = {1, 2, 4}
        if value not in valid_values:
            raise ValueError(f"micro_batch_size must be one of {valid_values}")
        return value

    @field_validator("tensor_model_parallel_size")
    def validate_tensor_model_parallel_size(cls, value):
        valid_values = {4, 8, 16}
        if value not in valid_values:
            raise ValueError(f"tensor_model_parallel_size must be one of {valid_values}")
        return value

    @field_validator("pipeline_model_parallel_size")
    def validate_pipeline_model_parallel_size(cls, value):
        valid_values = {2, 4, 8}
        if value not in valid_values:
            raise ValueError(f"pipeline_model_parallel_size must be one of {valid_values}")
        return value


class TrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_limit: str = "3:00:00"
    name: str = "run"


class Training(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: Literal["gpt3/40b_improved", "llama/llama2_70b"] = Field(default="gpt3/40b_improved")
    exp_manager: ExpManager = Field(default_factory=ExpManager)
    trainer: Trainer = Field(default_factory=Trainer)
    model: TrainingModel = Field(default_factory=TrainingModel)
    run: TrainingRun = Field(default_factory=TrainingRun)


class NeMoLauncherCmdArgs(CmdArgs):
    """NeMoLauncher test command arguments."""

    repository_url: str = Field(default="https://github.com/NVIDIA/NeMo-Framework-Launcher.git")
    repository_commit_hash: str = Field(default="cf411a9ede3b466677df8ee672bcc6c396e71e1a")
    docker_image_url: str = Field(default="nvcr.io/nvidian/nemofw-training:24.01.01")
    stages: str = Field(default='["training"]')
    data_dir: str = Field(default="DATA_DIR")
    numa_mapping: NumaMapping = Field(default_factory=NumaMapping)
    cluster: Cluster = Field(default_factory=Cluster)
    training: Training = Field(default_factory=Training)


class NeMoLauncherTestDefinition(TestDefinition):
    """Test object for NeMoLauncher."""

    cmd_args: NeMoLauncherCmdArgs
