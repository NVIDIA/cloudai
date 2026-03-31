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
from typing import Optional, cast

from pydantic import ConfigDict, Field

from cloudai.core import GitRepo, Installable, JobStatusResult, System, TestRun
from cloudai.models.workload import CmdArgs
from cloudai.workloads.common.llm_serving import (
    LLMServingArgs,
    LLMServingCmdArgs,
    LLMServingTestDefinition,
    all_gpu_ids,
    calculate_decode_gpu_ids,
    calculate_prefill_gpu_ids,
)

VLLM_SERVE_LOG_FILE = "vllm-serve.log"
VLLM_BENCH_LOG_FILE = "vllm-bench.log"
VLLM_BENCH_JSON_FILE = "vllm-bench.json"
VLLM_STANDALONE_BOOL_FLAGS = {"aggregate-engine-logging", "disable-log-stats", "grpc", "headless"}


class VllmArgs(LLMServingArgs):
    """Base command arguments for vLLM instances."""

    nixl_threads: int | list[int] | None = Field(
        default=None,
        description="Set ``kv_connector_extra_config.num_threads`` for ``--kv-transfer-config`` CLI argument.",
    )

    @property
    def serve_args_exclude(self) -> set[str]:
        return super().serve_args_exclude | {"nixl_threads"}

    @property
    def serve_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.model_dump(exclude=self.serve_args_exclude, exclude_none=True).items():
            opt = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{opt}")
                elif opt not in VLLM_STANDALONE_BOOL_FLAGS:
                    args.append(f"--no-{opt}")
            elif value == "":
                args.append(f"--{opt}")
            else:
                args.extend([f"--{opt}", str(value)])
        return args


class VllmCmdArgs(LLMServingCmdArgs[VllmArgs]):
    """vLLM serve command arguments."""

    model_config = ConfigDict(extra="forbid")  # arbitrary fields are allowed per decode/prefill, not here

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


class VllmTestDefinition(LLMServingTestDefinition[VllmCmdArgs]):
    """Test object for vLLM."""

    bench_cmd_args: VllmBenchCmdArgs = VllmBenchCmdArgs()
    proxy_script_repo: GitRepo | None = None

    @property
    def extra_installables(self) -> list[Installable]:
        installables: list[Installable] = []
        if self.proxy_script_repo:
            installables.append(self.proxy_script_repo)
        return installables

    @staticmethod
    def _validate_vllm_parallelism_constraints(role: str, args: VllmArgs, gpu_count: int) -> bool:
        tp = cast(int, getattr(args, "tensor_parallel_size", 1))
        pp = cast(int, getattr(args, "pipeline_parallel_size", 1))
        dp = cast(int, getattr(args, "data_parallel_size", 1))
        ep_enabled = cast(bool, getattr(args, "expert_parallel", getattr(args, "enable_expert_parallel", False)))
        all2all_backend = cast(str, getattr(args, "all2all_backend", ""))

        constraint1 = (tp * pp * dp) <= gpu_count
        if not constraint1:
            logging.error(
                "vLLM %s constraint failed: (tp * pp * dp) <= num_gpus. tp=%s pp=%s dp=%s num_gpus=%s",
                role,
                tp,
                pp,
                dp,
                gpu_count,
            )
            return False

        using_flashinfer_all2allv = all2all_backend == "flashinfer_all2allv"
        constraint2 = not (using_flashinfer_all2allv and dp > 1 and ep_enabled)
        if not constraint2:
            logging.error(
                "vLLM %s constraint failed: flashinfer_all2allv only works with DP=1, or with DP>1 and expert "
                "parallel disabled. all2all_backend=%s dp=%s expert_parallel=%s",
                role,
                all2all_backend,
                dp,
                ep_enabled,
            )
            return False

        return True

    def constraint_check(self, tr: TestRun, system: Optional[System]) -> bool:
        system_gpus_per_node = getattr(system, "gpus_per_node", None) if system is not None else None
        num_nodes = tr.nnodes

        if self.cmd_args.prefill is None:
            return self._validate_vllm_parallelism_constraints(
                role="decode",
                args=self.cmd_args.decode,
                gpu_count=len(all_gpu_ids(self, system_gpus_per_node)),
            )

        return self._validate_vllm_parallelism_constraints(
            role="prefill",
            args=self.cmd_args.prefill,
            gpu_count=len(calculate_prefill_gpu_ids(self, num_nodes, system_gpus_per_node)),
        ) and self._validate_vllm_parallelism_constraints(
            role="decode",
            args=self.cmd_args.decode,
            gpu_count=len(calculate_decode_gpu_ids(self, num_nodes, system_gpus_per_node)),
        )

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
