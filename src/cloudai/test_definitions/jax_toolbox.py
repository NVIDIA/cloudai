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

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_serializer, root_validator

from cloudai import CmdArgs, TestDefinition


class JaxFdl(BaseModel):
    """JAX FDL configuration."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_policy: Union[str, list[str]] = "save_nothing"
    dcn_mesh_shape: Union[str, list[str]] = "'[1, 1, 1]'"
    fprop_dtype: Union[str, list[str]] = "bfloat16"
    ep: Union[int, list[int]] = 1
    tp: Union[int, list[int]] = 1
    fsdp: Union[int, list[int]] = 1
    dp: Union[int, list[int]] = 1
    ici_mesh_shape: Union[str, list[str]] = "'[1, 8, 1]'"
    max_steps: Union[int, list[int]] = 20
    num_gpus: Union[int, list[int]] = 64
    num_microbatches: Union[int, list[int]] = 1
    num_stages: Union[int, list[int]] = 1
    percore_batch_size: Union[float, list[float]] = 4
    use_fp8: Union[int, list[int]] = 1
    use_repeated_layer: Union[bool, list[bool]] = False

    @field_serializer("fprop_dtype")
    def fprop_dtype_serializer(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        return f'\\"{value}\\"'

    @field_serializer("checkpoint_policy")
    def checkpoint_policy_serializer(self, value: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(value, list):
            return [self._serialize_single_policy(v) for v in value]
        return self._serialize_single_policy(value)

    def _serialize_single_policy(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        elif value.startswith('"') and value.endswith('"'):
            return value.replace('"', '\\"')
        return f'\\"{value}\\"'

    @root_validator(pre=True)
    def check_mesh_and_parallelism(cls, values):
        ep = values.get("ep")
        tp = values.get("tp")
        fsdp = values.get("fsdp")
        dp = values.get("dp")
        ici_mesh_shape = values.get("ici_mesh_shape")
        dcn_mesh_shape = values.get("dcn_mesh_shape")

        parallelism_defined = ep is not None or tp is not None or fsdp is not None or dp is not None
        mesh_defined = ici_mesh_shape is not None or dcn_mesh_shape is not None

        if parallelism_defined and mesh_defined:
            raise ValueError(
                "Cannot define both parallelism (ep, tp, fsdp, dp) and mesh shapes (ici_mesh_shape, dcn_mesh_shape)."
            )
        elif not parallelism_defined and not mesh_defined:
            raise ValueError(
                "Must define either parallelism (ep, tp, fsdp, dp) or mesh shapes (ici_mesh_shape, dcn_mesh_shape)."
            )

        return values


class JaxToolboxCmdArgs(CmdArgs):
    """JAX Toolbox test command arguments."""

    docker_image_url: str
    load_container: bool = True
    output_path: Optional[str] = None


class JaxToolboxTestDefinition(TestDefinition):
    """Test object for JAX Toolbox."""

    pass


class XLAFlags(BaseModel):
    """XLA flags configuration."""

    model_config = ConfigDict(extra="forbid")

    xla_disable_hlo_passes: Union[str, list[str]] = "rematerialization"
    xla_dump_hlo_pass_re: Union[str, list[str]] = ".*"
    xla_gpu_enable_all_gather_combine_by_dim: Union[bool, list[bool]] = False
    xla_gpu_enable_highest_priority_async_stream: Union[bool, list[bool]] = True
    xla_gpu_enable_latency_hiding_scheduler: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_all_gather: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_all_reduce: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_reduce_scatter: Union[bool, list[bool]] = True
    xla_gpu_enable_reduce_scatter_combine_by_dim: Union[bool, list[bool]] = False
    xla_gpu_enable_triton_gemm: Union[bool, list[bool]] = False
    xla_gpu_enable_triton_softmax_fusion: Union[bool, list[bool]] = False
    xla_gpu_enable_while_loop_double_buffering: Union[bool, list[bool]] = True
    xla_gpu_graph_level: Union[int, list[int]] = 0


class SetupFlags(BaseModel):
    """Setup flags configuration."""

    model_config = ConfigDict(extra="forbid")

    docker_workspace_dir: str = "/opt/paxml/workspace/"
    enable_checkpoint_saving: bool = False
    gpus_per_node: int = 8
    mpi: str = "pmix"
    num_nodes: int = 8
    tfds_data_dir: str = "/opt/dataset"
