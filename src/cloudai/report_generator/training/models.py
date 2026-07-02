# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data models for training parsers."""

from collections.abc import Hashable
from dataclasses import MISSING, dataclass, fields
from typing import Any, List, Optional


@dataclass(frozen=True)
class Scalar:
    """A single scalar event from a training run (source-agnostic: TensorBoard today, others later)."""

    tag: str
    step: int
    value: float
    wall_time: float

    @classmethod
    def from_record(cls, record: dict[Hashable, Any]) -> "Scalar":
        """Build from a {column: value} record (e.g. a tbparse DataFrame row)."""
        return cls(tag=record["tag"], step=record["step"], value=record["value"], wall_time=record["wall_time"])


@dataclass(kw_only=True)
class TrainingStep:
    """Results for a single training iteration."""

    iteration: int
    step_time_sec: float
    loss: float
    memory_reserved_bytes: float
    memory_allocated_bytes: float
    tflops_per_gpu: Optional[float] = None  # NeMo FLOPs could be missing for some models


OPTIONAL_STEP_FIELDS = {f.name for f in fields(TrainingStep) if f.default is not MISSING}


@dataclass(kw_only=True)
class TrainingConfig:
    """
    Resolved training configuration from the framework artifact + CloudAI.

    The CloudAI-computed fields (test_template_name, data_parallel_size, model_name, world_size, num_nodes) default
    here and are filled in by the parser after construction.
    """

    # Batch
    micro_batch_size: int
    global_batch_size: int
    seq_length: int

    # Parallelism
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: Optional[int]
    virtual_pipeline_parallel_size: Optional[int]
    sequence_parallel: bool
    expert_parallel_size: int
    data_parallel_size: Optional[int] = None  # CloudAI-computed (None when gpus_per_node is unavailable)

    # Model architecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int
    kv_channels: int
    normalization: str
    position_embedding_type: str
    model_name: str = ""  # CloudAI-computed

    # MoE
    num_experts: Optional[int]
    moe_router_topk: Optional[int]
    moe_ffn_hidden_size: Optional[int]
    moe_grouped_gemm: Optional[bool]

    # Hardware
    world_size: Optional[int] = None  # CloudAI-computed (None when gpus_per_node is unavailable)
    num_nodes: int = 0  # CloudAI-computed

    # Profiling (CloudAI-computed from the run's nsys/profiler settings)
    profiling_enabled: bool = False
    profiling_start_step: Optional[int] = None
    profiling_stop_step: Optional[int] = None

    # Identity
    test_template_name: str = ""  # CloudAI-computed


@dataclass
class TrainingResults:
    """Container for parsed training output."""

    config: TrainingConfig
    steps: List[TrainingStep]
