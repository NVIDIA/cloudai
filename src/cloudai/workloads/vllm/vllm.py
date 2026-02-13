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

from pydantic import ConfigDict, Field, model_validator

from cloudai.core import DockerImage, GitRepo, HFModel, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

VLLM_SERVE_LOG_FILE = "vllm-serve.log"
VLLM_BENCH_LOG_FILE = "vllm-bench.log"
VLLM_BENCH_JSON_FILE = "vllm-bench.json"


class VllmArgs(CmdArgs):
    """Base command arguments for vLLM instances."""

    gpu_ids: str | list[str] | None = Field(
        default=None, description="Comma-separated GPU IDs. If not set, will use all available GPUs."
    )

    nixl_threads: int | list[int] | None = Field(
        default=None,
        description="Set ``kv_connector_extra_config.num_threads`` for ``--kv-transfer-config`` CLI argument.",
    )

    @property
    def serve_args(self) -> list[str]:
        """Convert cmd_args_dict to command-line arguments list for vllm serve."""
        args = []
        for k, v in self.model_dump(exclude={"gpu_ids"}, exclude_none=True).items():
            opt = f"--{k.replace('_', '-')}"
            if v == "":
                args.append(opt)
            else:
                args.extend([opt, str(v)])
        return args


class VllmCmdArgs(CmdArgs):
    """vLLM serve command arguments."""

    model_config = ConfigDict(extra="forbid")  # arbitrary fields are allowed per decode/prefill, not here

    docker_image_url: str
    port: int = 8000
    vllm_serve_wait_seconds: int = 300
    proxy_script: str = "/opt/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"

    model: str = "Qwen/Qwen3-0.6B"
    prefill: VllmArgs | None = Field(
        default=None,
        description="Prefill instance arguments. If not set, a single instance without disaggregation will be used.",
    )
    decode: VllmArgs = Field(default_factory=VllmArgs, description="Decode instance arguments.")


class VllmBenchCmdArgs(CmdArgs):
    """vLLM bench serve command arguments."""

    random_input_len: int = 16
    random_output_len: int = 128
    max_concurrency: int = 16
    num_prompts: int = 30


class VllmTestDefinition(TestDefinition):
    """Test object for vLLM."""

    cmd_args: VllmCmdArgs
    bench_cmd_args: VllmBenchCmdArgs = VllmBenchCmdArgs()
    proxy_script_repo: GitRepo | None = None

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
        installables = [*self.git_repos, self.docker_image, self.hf_model]
        if self.proxy_script_repo:
            installables.append(self.proxy_script_repo)
        return installables

    @model_validator(mode="after")
    def check_gpu_ids_setup(self) -> VllmTestDefinition:
        if self.cmd_args.prefill:
            prefill_set = bool(self.cmd_args.prefill.gpu_ids)
            decode_set = bool(self.cmd_args.decode.gpu_ids)
            if prefill_set != decode_set:
                raise ValueError("Both prefill and decode gpu_ids must be set or both must be None.")
        return self

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        log_path = tr.output_path / VLLM_BENCH_LOG_FILE
        if not log_path.is_file():
            return JobStatusResult(is_successful=False, error_message=f"vLLM bench log not found in {tr.output_path}.")

        has_results_marker = False
        with log_path.open("r") as f:
            for line in f:
                if "============ Serving Benchmark Result ============" in line:
                    has_results_marker = True
                    continue
                if has_results_marker and "Successful requests:" in line:
                    try:
                        num_successful_requests = int(line.split()[2])
                        if num_successful_requests > 0:
                            return JobStatusResult(is_successful=True)
                    except Exception as e:
                        logging.debug(f"Error parsing number of successful requests: {e}")

        return JobStatusResult(
            is_successful=False, error_message=f"vLLM bench log does not contain benchmark result in {tr.output_path}."
        )
