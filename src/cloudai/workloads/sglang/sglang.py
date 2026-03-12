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

import json
import logging

from pydantic import ConfigDict, Field, model_validator

from cloudai.core import DockerImage, HFModel, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

SGLANG_SERVE_LOG_FILE = "sglang-serve.log"
SGLANG_BENCH_LOG_FILE = "sglang-bench.log"
SGLANG_BENCH_JSONL_FILE = "sglang-bench.jsonl"


class SglangArgs(CmdArgs):
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
    def serve_args(self) -> list[str]:
        """Convert cmd_args_dict to command-line arguments list for SGLang serve command."""
        args: list[str] = []
        for key, value in self.model_dump(
            exclude={"gpu_ids", "disaggregation_transfer_backend"}, exclude_none=True
        ).items():
            opt = f"--{key.replace('_', '-')}"
            if value == "":
                args.append(opt)
            else:
                args.extend([opt, str(value)])
        return args


class SglangCmdArgs(CmdArgs):
    """SGLang serve command arguments."""

    model_config = ConfigDict(extra="forbid")

    docker_image_url: str
    model: str = "Qwen/Qwen3-8B"
    port: int = 8000
    serve_wait_seconds: int = 300
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


class SglangTestDefinition(TestDefinition):
    """Test object for SGLang."""

    cmd_args: SglangCmdArgs
    bench_cmd_args: SglangBenchCmdArgs = SglangBenchCmdArgs()

    _docker_image: DockerImage | None = None
    _hf_model: HFModel | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            self._hf_model = HFModel(model_name=self.cmd_args.model)
        return self._hf_model

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image, self.hf_model]

    @model_validator(mode="after")
    def check_gpu_ids_setup(self) -> SglangTestDefinition:
        if self.cmd_args.prefill:
            prefill_set = bool(self.cmd_args.prefill.gpu_ids)
            decode_set = bool(self.cmd_args.decode.gpu_ids)
            if prefill_set != decode_set:
                raise ValueError("Both prefill and decode gpu_ids must be set or both must be None.")
        return self

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        jsonl_path = tr.output_path / SGLANG_BENCH_JSONL_FILE
        if not jsonl_path.is_file():
            return JobStatusResult(
                is_successful=False, error_message=f"SGLang bench jsonl not found in {tr.output_path}."
            )

        completed_requests: int | None = None
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed_requests = int(record.get("completed"))
                break
            except Exception as e:
                logging.debug(f"Skipping invalid JSONL record in SGLang benchmark output: {e}")

        if completed_requests is not None and completed_requests > 0:
            return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=f"SGLang bench jsonl does not contain successful requests in {tr.output_path}.",
        )
