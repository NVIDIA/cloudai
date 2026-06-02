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

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cloudai.core import Installable, JobStatusResult, PythonEnvironment, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class MockerEngineArgs(BaseModel):
    """
    Engine simulation parameters for the Dynamo Mocker.

    Any additional dynamo.mocker engine flag can be added directly to
    [cmd_args.engine] in the TOML and will be forwarded to the dynamo.mocker
    CLI automatically via the extra="allow" passthrough (--engine-<key>).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    speedup_ratio: float = Field(default=1.0, gt=0)
    block_size: int = Field(default=64, gt=0)
    num_gpu_blocks_override: int = Field(default=16384, gt=0, description="Number of simulated KV cache blocks")
    enable_prefix_caching: bool = True

    # Simulates nixl KV block migration bandwidth (GB/s).
    # Only active in prefill_decode mode; ignored otherwise.
    kv_transfer_bandwidth: Union[float, List[float]] = Field(
        default=200.0, description="Simulated KV transfer bandwidth (GB/s)"
    )

    @field_validator("kv_transfer_bandwidth")
    @classmethod
    def kv_bandwidth_positive(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        vals = v if isinstance(v, list) else [v]
        if any(x <= 0 for x in vals):
            raise ValueError("kv_transfer_bandwidth values must be > 0")
        return v


class MockerWorkerInstanceArgs(BaseModel):
    """
    Named + extra flags forwarded to a single mocker instance (prefill or decode).

    Any field added here becomes --<prefix>-args-<key> in dynamo_mocker.sh.
    Example: max_num_seqs = 4 → --prefill-args-max-num-seqs 4 for prefill instances.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class MockerWorkerInstance(BaseModel):
    """
    Configuration for one worker role (prefill or decode).

    Mirrors the ai_dynamo WorkerConfig pattern:
      num_nodes                — number of mocker instances to launch for this role
      cmd                      — command to launch the worker (empty = use shell default)
      worker_initialized_regex — log-line pattern that signals the worker is ready
      extra_args                — raw CLI string appended verbatim to the mocker command
      args                     — named + extra flags forwarded as --<prefix>-args-<key>
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: int = Field(default=1, gt=0, description="Number of mocker instances for this role")
    cmd: str = Field(
        default="",
        description="Command to launch this worker. Empty = shell default (venv python -m dynamo.mocker).",
    )
    worker_initialized_regex: str = "created and running"
    extra_args: Optional[str] = None
    args: MockerWorkerInstanceArgs = Field(default_factory=MockerWorkerInstanceArgs)


class MockerWorkerConfig(BaseModel):
    """
    Worker topology for the Dynamo Mocker.

    disaggregation_mode controls the deployment topology:
      "none"           → single mocker process (combined prefill+decode)
      "prefill_decode" → separate prefill + decode mocker instances

    prefill_worker / decode_worker mirror the ai_dynamo WorkerConfig nesting:
      each holds worker_initialized_regex, optional extra_args (raw CLI string),
      and an args sub-section with named + extra flags forwarded as
      --prefill-args-<key> / --decode-args-<key>.

    Any additional dynamo.mocker topology flag can be added directly to
    [cmd_args.worker] in the TOML and will be forwarded via --mocker-<key>.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    disaggregation_mode: Literal["none", "prefill_decode"] = "none"
    num_workers: Union[int, List[int]] = Field(default=1)  # combined mode: simulated workers in one process

    # Disaggregated mode: per-role instance counts and commands live inside each worker instance.
    prefill_worker: MockerWorkerInstance = Field(default_factory=MockerWorkerInstance)
    decode_worker: MockerWorkerInstance = Field(default_factory=MockerWorkerInstance)

    @field_validator("num_workers")
    @classmethod
    def worker_counts_positive(cls, v: Union[int, List[int]]) -> Union[int, List[int]]:
        vals = v if isinstance(v, list) else [v]
        if any(x <= 0 for x in vals):
            raise ValueError("worker count values must be > 0")
        return v


class MockerFrontendArgs(BaseModel):
    """
    Frontend / ingress configuration for the Dynamo Mocker.

    Any additional dynamo.frontend flag can be added directly to
    [cmd_args.frontend] in the TOML and will be forwarded to dynamo.frontend
    via the extra="allow" passthrough (--frontend-<key>).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    http_port: int = Field(default=8000, ge=1, le=65535)
    router_mode: Literal["round_robin", "kv_router"] = "round_robin"


class MockerGenAIPerfArgs(BaseModel):
    """
    Benchmark parameters passed to genai-perf.

    cmd mirrors the ai_dynamo pattern — override the genai-perf launch command.
    Empty (default) means genai_perf.sh uses its built-in resolved binary.
    extra_args is a raw CLI string appended verbatim after all other flags.

    Any additional genai-perf flag (e.g. endpoint_type, streaming) can be
    added directly to [cmd_args.genai_perf] in the TOML and will be forwarded
    to the genai-perf CLI automatically via the extra="allow" passthrough.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    cmd: str = Field(
        default="",
        description="Override genai-perf binary. 'profile' is always appended. Empty = shell default.",
    )
    extra_args: Optional[str] = None
    input_tokens: Union[int, List[int]] = Field(default=5000)
    output_tokens: Union[int, List[int]] = Field(default=500)
    request_count: int = Field(default=1000, gt=0)
    replay_concurrency: Union[int, List[int]] = Field(default=100)
    # offline → --concurrency (throughput test)
    # online  → --request-rate (latency-under-load test)
    replay_mode: Literal["offline", "online"] = "offline"

    @field_validator("input_tokens", "output_tokens", "replay_concurrency")
    @classmethod
    def token_counts_positive(cls, v: Union[int, List[int]]) -> Union[int, List[int]]:
        vals = v if isinstance(v, list) else [v]
        if any(x <= 0 for x in vals):
            raise ValueError("values must be > 0")
        return v


class MockerAIPerfArgs(BaseModel):
    """
    Benchmark parameters passed to aiperf.

    cmd mirrors the ai_dynamo pattern — override the aiperf launch command.
    Empty (default) means aiperf.sh uses its built-in resolved binary.
    extra_args is a raw CLI string appended verbatim after all other flags.

    Base fields mirror MockerGenAIPerfArgs for easy tool switching.
    Any additional aiperf-specific flag (e.g. arrival_pattern, benchmark_duration)
    can be added directly to [cmd_args.aiperf] in the TOML and will be forwarded
    to the aiperf CLI automatically via the extra="allow" passthrough.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    cmd: str = Field(
        default="",
        description="Override aiperf binary. 'profile' is always appended. Empty = shell default.",
    )
    extra_args: Optional[str] = None
    input_tokens: Union[int, List[int]] = Field(default=5000)
    output_tokens: Union[int, List[int]] = Field(default=500)
    request_count: int = Field(default=1000, gt=0)
    replay_concurrency: Union[int, List[int]] = Field(default=100)
    # offline → --concurrency (throughput test)
    # online  → --request-rate (latency-under-load test)
    replay_mode: Literal["offline", "online"] = "offline"

    @field_validator("input_tokens", "output_tokens", "replay_concurrency")
    @classmethod
    def token_counts_positive(cls, v: Union[int, List[int]]) -> Union[int, List[int]]:
        vals = v if isinstance(v, list) else [v]
        if any(x <= 0 for x in vals):
            raise ValueError("values must be > 0")
        return v


class DynamoMockerCmdArgs(CmdArgs):
    """Top-level command arguments for the Dynamo Mocker workload."""

    model_path: str = "Qwen/Qwen3-0.6B"
    nats_cmd: str = Field(
        default="nats-server -js",
        description="Command used to launch nats-server. Override with a full path if nats-server is not on PATH.",
    )
    engine: MockerEngineArgs = Field(default_factory=MockerEngineArgs)
    worker: MockerWorkerConfig = Field(default_factory=MockerWorkerConfig)
    frontend: MockerFrontendArgs = Field(default_factory=MockerFrontendArgs)
    benchmark_tool: Literal["genai_perf", "aiperf"] = "genai_perf"
    genai_perf: MockerGenAIPerfArgs = Field(default_factory=MockerGenAIPerfArgs)
    aiperf: MockerAIPerfArgs = Field(default_factory=MockerAIPerfArgs)


class DynamoMockerTestDefinition(TestDefinition):
    """
    Test definition for the Dynamo Mocker workload.

    Runs dynamo.mocker + dynamo.frontend as a lightweight GPU-free LLM inference
    simulator, then benchmarks it with genai-perf or aiperf via dynamo_mocker.sh.
    Select the benchmark tool with cmd_args.benchmark_tool ("genai_perf" or "aiperf").

    Supports two topologies mirroring ai_dynamo.sh:
      - Combined (worker.disaggregation_mode=none): single mocker handles prefill+decode
      - Disaggregated (worker.disaggregation_mode=prefill_decode): separate prefill and
        decode mocker instances with KV event publishing and simulated transfer

    Requires: ai-dynamo and genai-perf or aiperf pip-installed in the active environment.
    """

    cmd_args: DynamoMockerCmdArgs

    success_marker: str = "success-marker.txt"
    failure_marker: str = "failure-marker.txt"

    _python_environment: Optional[PythonEnvironment] = None

    @property
    def python_environment(self) -> PythonEnvironment:
        if not self._python_environment:
            self._python_environment = PythonEnvironment(
                name="dynamo-mocker",
                python_version="3.12",
                requirements=["ai-dynamo", "genai-perf", "aiperf"],
            )
        return self._python_environment

    @property
    def installables(self) -> list[Installable]:
        return [self.python_environment]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        failure_marker = output_path / self.failure_marker
        success_marker = output_path / self.success_marker

        if failure_marker.exists():
            contents = failure_marker.read_text(encoding="utf-8").strip()
            return JobStatusResult(is_successful=False, error_message=f"Failure marker found: {contents}")

        if not success_marker.exists():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Success marker not found: {success_marker.absolute()}. "
                    "Check stdout.txt and stderr.txt for errors."
                ),
            )

        report_csv = output_path / "benchmark_report.csv"
        if not report_csv.exists():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"benchmark_report.csv not found in {output_path}. The benchmark may not have completed."
                ),
            )

        return JobStatusResult(is_successful=True)
