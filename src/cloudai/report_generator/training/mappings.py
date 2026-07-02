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

"""
Per-workload field mappings for the training report parsers.

TrainingStep fields map to TensorBoard scalar tags; TrainingConfig fields map to framework artifact field names.
"""

# TrainingStep field -> TB scalar tag, read from the value column. iteration is the step
# axis (the row key); per-field unit scale lives in *_SCALE below.
NEMO_STEPS: dict[str, str] = {
    "step_time_sec": "train_step_timing in s",
    "loss": "reduced_train_loss",
    "memory_reserved_bytes": "max_memory_reserved",
    "memory_allocated_bytes": "max_memory_allocated",
    "tflops_per_gpu": "TFLOPS_per_GPU",
}

MEGATRON_STEPS: dict[str, str] = {
    "step_time_sec": "iteration-time",
    "loss": "lm loss",
    "memory_reserved_bytes": "mem-reserved-bytes",
    "memory_allocated_bytes": "mem-allocated-bytes",
    "tflops_per_gpu": "throughput",
}

MEGATRON_BRIDGE_STEPS: dict[str, str] = {
    "step_time_sec": "iteration-time",
    "loss": "lm loss",
    "memory_reserved_bytes": "memory/mem-reserved-gigabytes",
    "memory_allocated_bytes": "memory/mem-allocated-gigabytes",
    "tflops_per_gpu": "throughput/tflops/device",
}

MEGATRON_BRIDGE_SCALE: dict[str, float] = {
    "memory_reserved_bytes": 1e9,
    "memory_allocated_bytes": 1e9,
}


# Framework's resolved config artifact. (world_size, num_nodes, model_name) and computed data_parallel_size are not
# mapped here.
NEMO_MODEL_CONFIG: dict[str, str] = {
    "micro_batch_size": "data.micro_batch_size",
    "global_batch_size": "data.global_batch_size",
    "seq_length": "data.seq_length",
    "tensor_parallel_size": "parallelism.tensor_model_parallel_size",
    "pipeline_parallel_size": "parallelism.pipeline_model_parallel_size",
    "context_parallel_size": "parallelism.context_parallel_size",
    "virtual_pipeline_parallel_size": "parallelism.virtual_pipeline_model_parallel_size",
    "sequence_parallel": "parallelism.sequence_parallel",
    "expert_parallel_size": "parallelism.expert_model_parallel_size",
    "num_layers": "model.num_layers",
    "hidden_size": "model.hidden_size",
    "num_attention_heads": "model.num_attention_heads",
    "num_query_groups": "model.num_query_groups",
    "ffn_hidden_size": "model.ffn_hidden_size",
    "kv_channels": "model.kv_channels",
    "normalization": "model.normalization",
    "position_embedding_type": "model.position_embedding_type",
    "num_experts": "model.num_moe_experts",
    "moe_router_topk": "model.moe_router_topk",
    "moe_ffn_hidden_size": "model.moe_ffn_hidden_size",
    "moe_grouped_gemm": "model.moe_grouped_gemm",
}

MEGATRON_MODEL_CONFIG: dict[str, str] = {
    "micro_batch_size": "micro_batch_size",
    "global_batch_size": "global_batch_size",
    "seq_length": "seq_length",
    "tensor_parallel_size": "tensor_model_parallel_size",
    "pipeline_parallel_size": "pipeline_model_parallel_size",
    "context_parallel_size": "context_parallel_size",
    "virtual_pipeline_parallel_size": "virtual_pipeline_model_parallel_size",
    "sequence_parallel": "sequence_parallel",
    "expert_parallel_size": "expert_model_parallel_size",
    "num_layers": "num_layers",
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_query_groups": "num_query_groups",
    "ffn_hidden_size": "ffn_hidden_size",
    "kv_channels": "kv_channels",
    "normalization": "normalization",
    "position_embedding_type": "position_embedding_type",
    "num_experts": "num_experts",
    "moe_router_topk": "moe_router_topk",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
    "moe_grouped_gemm": "moe_grouped_gemm",
}

MEGATRON_BRIDGE_MODEL_CONFIG: dict[str, str] = {
    "micro_batch_size": "train.micro_batch_size",
    "global_batch_size": "train.global_batch_size",
    "seq_length": "model.seq_length",
    "tensor_parallel_size": "model.tensor_model_parallel_size",
    "pipeline_parallel_size": "model.pipeline_model_parallel_size",
    "context_parallel_size": "model.context_parallel_size",
    "virtual_pipeline_parallel_size": "model.virtual_pipeline_model_parallel_size",
    "sequence_parallel": "model.sequence_parallel",
    "expert_parallel_size": "model.expert_model_parallel_size",
    "num_layers": "model.num_layers",
    "hidden_size": "model.hidden_size",
    "num_attention_heads": "model.num_attention_heads",
    "num_query_groups": "model.num_query_groups",
    "ffn_hidden_size": "model.ffn_hidden_size",
    "kv_channels": "model.kv_channels",
    "normalization": "model.normalization",
    "position_embedding_type": "model.position_embedding_type",
    "num_experts": "model.num_moe_experts",
    "moe_router_topk": "model.moe_router_topk",
    "moe_ffn_hidden_size": "model.moe_ffn_hidden_size",
    "moe_grouped_gemm": "model.moe_grouped_gemm",
}


# CloudAI TestDefinition (user TOML + defaults). TrainingConfig field -> dotted path in TestDefinition.model_dump().
NEMO_TEST_CONFIG: dict[str, str] = {
    "profiling_enabled": "nsys.enable",
    "profiling_start_step": "extra_cmd_args.*start_step",
    "profiling_stop_step": "extra_cmd_args.*end_step",
}

MEGATRON_TEST_CONFIG: dict[str, str] = {
    "profiling_enabled": "nsys.enable",
    "profiling_start_step": "cmd_args.profile_step_start",
    "profiling_stop_step": "cmd_args.profile_step_end",
}

MEGATRON_BRIDGE_TEST_CONFIG: dict[str, str] = {
    "profiling_enabled": "cmd_args.enable_nsys",
    "profiling_start_step": "cmd_args.profiling_start_step",
    "profiling_stop_step": "cmd_args.profiling_stop_step",
}
