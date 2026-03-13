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

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

from pydantic import ConfigDict, Field, model_validator

from cloudai.core import JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs
from cloudai.workloads.common.llm_serving import (
    LLMServingArgs,
    LLMServingBenchReport,
    LLMServingCmdArgs,
    LLMServingTestDefinition,
)

SGLANG_SERVE_LOG_FILE = "sglang-serve.log"
SGLANG_BENCH_LOG_FILE = "sglang-bench.log"
SGLANG_BENCH_JSONL_FILE = "sglang-bench.jsonl"


class SglangArgs(LLMServingArgs):
    """Base command arguments for SGLang instances."""

    gpu_ids: str | list[str] | None = Field(
        default=None, description="Comma-separated GPU IDs. If not set, will use all available GPUs."
    )

    disaggregation_transfer_backend: str | list[str] | None = Field(
        default=None,
        description=(
            "Transfer backend used in disaggregated mode. It is consumed by command generation and not emitted "
            "as a generic serve argument."
        ),
    )

    @property
    def serve_args_exclude(self) -> set[str]:
        return {"gpu_ids", "disaggregation_transfer_backend"}


class SglangCmdArgs(LLMServingCmdArgs):
    """SGLang serve command arguments."""

    model_config = ConfigDict(extra="forbid")

    docker_image_url: str
    model: str = "Qwen/Qwen3-8B"
    port: int = 8000
    health_endpoint: str = "/health"

    serve_module: str = "sglang.launch_server"
    router_module: str = "sglang_router.launch_router"
    bench_module: str = "sglang.bench_serving"

    prefill: SglangArgs | None = Field(
        default=None,
        description="Prefill instance arguments. If not set, a single instance without disaggregation is used.",
    )
    decode: SglangArgs = Field(default_factory=SglangArgs, description="Decode instance arguments.")


class SglangBenchCmdArgs(CmdArgs):
    """SGLang bench_serving command arguments."""

    backend: str = "sglang"
    dataset_name: str = "random"
    num_prompts: int = 30
    max_concurrency: int = 16
    random_input: int = 16
    random_output: int = 128
    warmup_requests: int = 2
    random_range_ratio: float = 1.0
    output_details: bool = True


class SglangTestDefinition(LLMServingTestDefinition[SglangCmdArgs]):
    """Test object for SGLang."""

    cmd_args: SglangCmdArgs
    bench_cmd_args: SglangBenchCmdArgs = SglangBenchCmdArgs()

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        res = parse_sglang_bench_output(tr.output_path / SGLANG_BENCH_JSONL_FILE)
        if res and res.completed > 0:
            return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=f"SGLang bench jsonl does not contain successful requests in {tr.output_path}.",
        )


class SGLangBenchReport(LLMServingBenchReport):
    """Parsed benchmark data from SGLang bench_serving output."""

    request_throughput: float

    @property
    def throughput(self) -> float:
        return self.request_throughput

    @model_validator(mode="before")
    @classmethod
    def derive_num_prompts(cls, data):
        if isinstance(data, dict) and "num_prompts" not in data:
            input_lens = data.get("input_lens")
            if isinstance(input_lens, list):
                data = dict(data)
                data["num_prompts"] = len(input_lens)
        return data


@cache
def parse_sglang_bench_output(jsonl_file: Path) -> SGLangBenchReport | None:
    """Parse SGLang benchmark output from JSONL file."""
    if not jsonl_file.is_file():
        return None

    with jsonl_file.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                parsed = SGLangBenchReport.model_validate_json(line)
                if parsed.completed <= 0:
                    return None
                return parsed
            except Exception as e:
                logging.debug(f"Skipping invalid JSONL record in SGLang benchmark output: {e}")
                continue

    return None
