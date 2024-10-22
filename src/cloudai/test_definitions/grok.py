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

from pydantic import ConfigDict, Field

from .jax_toolbox import JaxFdl, JaxToolboxCmdArgs, JaxToolboxTestDefinition, PreTest, SetupFlags, XLAFlags


class GrokFdl(JaxFdl):
    """FDL for Grok."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    checkpoint_policy: str = '"save_iteration_input"'
    combine_qkv: bool = False
    dcn_mesh_shape: str = "'[1, 8, 1, 1]'"
    dims_per_head: int = 128
    hidden_dims: int = 32768
    ici_mesh_shape: str = "'[1, 1, 8, 1]'"
    max_seq_len: int = 8192
    model_dims: int = 6144
    num_experts: int = 8
    num_groups: int = 64
    num_heads: int = 48
    num_kv_heads: int = 8
    num_layers: int = 64
    percore_batch_size: float = 1.0
    use_expert_parallel: bool = True
    use_fp8: int = 1
    use_te_dpa: bool = True
    vocab_size: int = 131072


class GrokPerfXLAFlags(XLAFlags):
    """Grok performance XLA flags."""

    combine_threshold_bytes: int = 301989888
    xla_gpu_run_post_layout_collective_pipeliner: bool = False
    xla_gpu_use_memcpy_local_p2p: bool = False
    xla_gpu_pgle_profile_file_or_directory_path: str = "/opt/paxml/workspace/pgle_output_profile.pbtxt"


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
    fdl_config: str = "paxml.tasks.lm.params.nvidia.Grok_Proxy"
    enable_pgle: bool = True
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
            if v is None:
                continue

            if k in {"profile", "perf"}:
                res.setdefault(f"Grok.{k}", {})
                res[f"Grok.{k}"]["XLA_FLAGS"] = v
            elif k in {"pre_test", "docker_image_url", "load_container", "output_path"}:
                res[k] = v
            else:
                res[f"Grok.{k}"] = v
        return res
