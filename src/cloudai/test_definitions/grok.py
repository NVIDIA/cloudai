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

from pydantic import ConfigDict, Field

from cloudai.test_definitions.jax_toolbox import XLAFlags

from .jax_toolbox import JaxFdl, JaxToolboxCmdArgs, JaxToolboxTestDefinition, NCCLPreTest, PreTest, SetupFlags


class GrokFdl(JaxFdl):
    """FDL for Grok."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    checkpoint_policy: str = '"save_iteration_input"'
    dcn_mesh_shape: str = "'[1, 8, 1, 1]'"
    ici_mesh_shape: str = "'[1, 1, 8, 1]'"
    percore_batch_size: float = 1.0  # type: ignore
    use_fp8: int = 1  # type: ignore
    combine_qkv: bool = False
    dims_per_head: int = 128
    hidden_dims: int = 32768
    max_seq_len: int = 8192
    model_dims: int = 6144
    num_experts: int = 8
    num_groups: int = 64
    num_heads: int = 48
    num_kv_heads: int = 8
    num_layers: int = 2
    use_expert_parallel: bool = True
    use_te_dpa: bool = True
    vocab_size: int = 131072


class GrokPerfXLAFlags(XLAFlags):
    """Grok performance XLA flags."""

    combine_threshold_bytes: int = 301989888
    xla_gpu_run_post_layout_collective_pipeliner: bool = False
    xla_gpu_use_memcpy_local_p2p: bool = False


class GrokProfileXLAFlags(XLAFlags):
    """Grok profile XLA flags."""

    xla_gpu_disable_async_collectives: str = (
        "ALLREDUCE,ALLGATHER,REDUCESCATTER,COLLECTIVEBROADCAST,ALLTOALL,COLLECTIVEPERMUTE"
    )
    xla_gpu_run_post_layout_collective_pipeliner: bool = False
    xla_gpu_enable_latency_hiding_scheduler: bool = False


class GrokCmdArgs(JaxToolboxCmdArgs):
    """Grok test command arguments."""

    fdl: GrokFdl = Field(default_factory=GrokFdl)
    fdl_config: Optional[str] = None
    enable_pgle: bool = False
    setup_flags: SetupFlags = Field(default_factory=SetupFlags)
    profile: GrokProfileXLAFlags = Field(default_factory=GrokProfileXLAFlags)
    perf: GrokPerfXLAFlags = Field(default_factory=GrokPerfXLAFlags)
    pre_test: PreTest = Field(default_factory=PreTest)


class GrokTestDefinition(JaxToolboxTestDefinition):
    """Test object for Grok."""

    cmd_args: GrokCmdArgs

    @property
    def cmd_args_dict(self):
        d = self.cmd_args.model_dump()
        res = {}
        for k, v in d.items():
            if not v:
                continue

            if k in {"profile", "perf"}:
                res.setdefault(f"Grok.{k}", {})
                res[f"Grok.{k}"]["XLA_FLAGS"] = v
            else:
                if k == "xla_flags":
                    k = k.upper()
                res[f"Grok.{k}"] = v
        return res
