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

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from cloudai import CmdArgs, TestDefinition
from cloudai.test_definitions.nccl import NCCLCmdArgs


class JaxFdl(BaseModel):
    """JAX FDL configuration."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_policy: str = "save_nothing"
    dcn_mesh_shape: str = "'[1, 1, 1]'"
    fprop_dtype: str = "bfloat16"
    ici_mesh_shape: str = "'[1, 8, 1]'"
    max_steps: int = 20
    num_gpus: int = 64
    num_microbatches: int = 1
    num_stages: int = 1
    percore_batch_size: float = 4
    use_fp8: int = 1
    use_repeated_layer: bool = False

    @field_serializer("fprop_dtype")
    def fprop_dtype_serializer(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        return f'\\"{value}\\"'

    @field_serializer("checkpoint_policy")
    def checkpoint_policy_serializer(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        elif value.startswith('"') and value.endswith('"'):
            return value.replace('"', '\\"')
        return f'\\"{value}\\"'


class NCCLCmdAgrsPreTest(NCCLCmdArgs):
    """NCCL pre-test command arguments."""

    num_nodes: int = 8
    stepfactor: int = 2
    minbytes: str = "8M"
    maxbytes: str = "16G"
    blocking: int = 1

    def model_post_init(self, _: Any) -> None:
        self.subtest_name = "all_gather_perf_mpi"
        self.docker_image_url = "nvcr.io/nvidia/pytorch:24.02-py3"


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

    docker_image_url: str
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
