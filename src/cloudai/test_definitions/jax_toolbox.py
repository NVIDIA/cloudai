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

from pydantic import BaseModel, ConfigDict, Field

from cloudai import CmdArgs, TestDefinition

from .nccl import NCCLCmdArgs


class JaxFdl(BaseModel):
    """JAX FDL configuration."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_policy: str = '"save_nothing"'
    dcn_mesh_shape: str = "'[1, 1, 1]'"
    fprop_dtype: str = '"bfloat16"'
    ici_mesh_shape: str = "'[1, 8, 1]'"
    max_steps: int = 20
    num_gpus: int = 64
    num_microbatches: int = 1
    num_stages: int = 1
    percore_batch_size: int = 4
    use_fp8: bool = False
    use_repeated_layer: bool = False


class NCCLCmdAgrsPreTest(NCCLCmdArgs):
    """NCCL pre-test command arguments."""

    num_nodes: int = 2


class PreTest(BaseModel):
    """Pre-test configuration."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    enable: bool = True
    nccl_test: NCCLCmdAgrsPreTest = Field(default_factory=NCCLCmdAgrsPreTest)


class NCCLPreTest(BaseModel):
    """Pre-test configuration."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    nccl_test: Optional[NCCLCmdAgrsPreTest] = None


class JaxToolboxCmdArgs(CmdArgs):
    """JAX Toolbox test command arguments."""

    load_container: bool = True


class JaxToolboxTestDefinition(TestDefinition):
    """Test object for JAX Toolbox."""

    pass


class XLAFlags(BaseModel):
    """XLA flags configuration."""

    model_config = ConfigDict(extra="forbid")

    xla_disable_hlo_passes: str = "rematerialization"
    xla_dump_hlo_pass_re: str = ".*"
    xla_gpu_enable_all_gather_combine_by_dim: bool = False
    xla_gpu_enable_highest_priority_async_stream: bool = True
    xla_gpu_enable_latency_hiding_scheduler: bool = True
    xla_gpu_enable_pipelined_all_gather: bool = True
    xla_gpu_enable_pipelined_all_reduce: bool = True
    xla_gpu_enable_pipelined_reduce_scatter: bool = True
    xla_gpu_enable_reduce_scatter_combine_by_dim: bool = False
    xla_gpu_enable_triton_gemm: bool = False
    xla_gpu_enable_triton_softmax_fusion: bool = False
    xla_gpu_enable_while_loop_double_buffering: bool = True
    xla_gpu_graph_level: int = 0


class SetupFlags(BaseModel):
    """Setup flags configuration."""

    model_config = ConfigDict(extra="forbid")

    docker_workspace_dir: str = "/opt/paxml/workspace/"
    enable_checkpoint_saving: bool = False
    gpus_per_node: int = 8
    mpi: str = "pmix"
    num_nodes: int = 8
    tfds_data_dir: str = "/opt/dataset"
