#!/usr/bin/env bash
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

set -euo pipefail
export INFMAX_CONTAINER_WORKSPACE=/infmax-workspace
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
install_agentic_deps

# Use the mounted model tokenizer without fetching it from the Hub.
export HF_HUB_CACHE="$(mktemp -d)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
tokenizer_cache="$HF_HUB_CACHE/models--Qwen--Qwen3-0.6B"
revision=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p "$tokenizer_cache/refs" "$tokenizer_cache/snapshots"
ln -s /model "$tokenizer_cache/snapshots/$revision"
printf '%s' "$revision" > "$tokenizer_cache/refs/main"

# Replay the same two pinned public test fixtures in each trial.
traces="$(mktemp -d)"
cp "$AIPERF_DIR/tests/fixtures/weka_traces/simple.json" "$traces/"
cp "$AIPERF_DIR/tests/fixtures/weka_traces/one_subagent.json" "$traces/"
export MODEL=Qwen/Qwen3-0.6B MODEL_PREFIX=qwen3 FRAMEWORK=sglang PRECISION=bf16
export DURATION=60 CONC=1 PORT=8000 RESULT_FILENAME=qwen_c1
export IS_MULTINODE=false DISAGG=false KV_OFFLOADING=none KV_OFFLOAD_BACKEND=''
export AIPERF_WARMUP_REQUESTS_PER_LANE=1 AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_EXTRA_INPUTS=ignore_eos:true
export AGENTIC_OUTPUT_DIR=/logs/agentic
result_dir="$AGENTIC_OUTPUT_DIR/conc_1"
mkdir -p "$result_dir"
TRACE_SOURCE_FLAG="--input-file $traces --custom-dataset-type weka_trace"
build_replay_cmd "$result_dir"
REPLAY_CMD="${REPLAY_CMD/--trajectory-start-min-ratio 0.25/--trajectory-start-min-ratio 0.0}"
REPLAY_CMD="${REPLAY_CMD/--trajectory-start-max-ratio 0.75/--trajectory-start-max-ratio 0.0}"
REPLAY_CMD+=' --tokenizer Qwen/Qwen3-0.6B'
run_agentic_replay_and_write_outputs "$result_dir"
