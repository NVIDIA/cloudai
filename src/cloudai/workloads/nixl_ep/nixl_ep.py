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

import json
from pathlib import Path, PurePosixPath
from typing import ClassVar, Optional

import toml
from pydantic import Field, field_validator, model_validator

from cloudai.core import DockerImage, GitRepo, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .log_parsing import parse_nixl_ep_bandwidth_samples


class NixlEPCmdArgs(CmdArgs):
    """Command line arguments for the NIXL Elastic EP benchmark."""

    docker_image_url: str = Field(description="URL of the Docker image that contains the NIXL EP benchmark.")
    elastic_script: str = Field(
        default="/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
        description=(
            "Path to the benchmark entrypoint, relative to the container's NIXL runtime root "
            "or absolute in the container."
        ),
    )
    python_executable: str = Field(default="python3", description="Python executable to use inside the container.")
    plan: str | list[str] | None = Field(
        default=None,
        description=(
            "Serialized phase plan to write into a per-run JSON file. "
            'Use a single string such as "[[0, 1], [0, 1, 2, 3]]".'
        ),
    )
    num_processes_per_node: int | list[int] = Field(
        description="Number of local worker processes to spawn on each allocated node.",
    )
    num_tokens: int = Field(default=128, ge=1, description="Tokens per dispatch.")
    num_experts_per_rank: int = Field(default=2, ge=1, description="Experts per rank.")
    hidden_dim: int = Field(default=7168, ge=1, description="Hidden dimension.")
    num_topk: int = Field(default=8, ge=1, description="Top-K routing value.")
    disable_ll_nvlink: bool = Field(
        default=False,
        description=(
            "Disable the benchmark's low-latency NVLink path. In the upstream NIXL EP example this also forces "
            "UCX to exclude CUDA IPC (`UCX_TLS=^cuda_ipc`), so it is best reserved for explicit RDMA-only "
            "comparisons rather than single-node bring-up."
        ),
    )
    kineto: bool = Field(default=False, description="Enable Kineto profiling.")
    debug_logging: bool = Field(
        default=False,
        description="Enable verbose NIXL EP/UCX logging and launcher-side diagnostics in the node logs.",
    )
    ucx_log_level: str | None = Field(
        default=None,
        description="Optional UCX log level override. Defaults to DEBUG when debug_logging is enabled.",
    )
    nixl_log_level: str | None = Field(
        default=None,
        description="Optional NIXL log level override. Defaults to TRACE when debug_logging is enabled.",
    )
    service_startup_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Seconds to wait for the master node's TCPStore and rank server to accept connections.",
    )
    rank_server_port: int = Field(default=10000, ge=1, le=65535, description="Rank server port.")
    store_port: int = Field(default=9999, ge=1, le=65535, description="TCPStore port.")

    @field_validator("num_processes_per_node", mode="after")
    @classmethod
    def validate_num_processes_per_node(cls, value: int | list[int]) -> int | list[int]:
        values = value if isinstance(value, list) else [value]
        if any(item < 1 for item in values):
            raise ValueError("num_processes_per_node must contain only positive integers")
        return value

    @model_validator(mode="after")
    def validate_plan_source(self) -> "NixlEPCmdArgs":
        if "input_json" in (self.model_extra or {}):
            raise ValueError("NixlEP does not accept `input_json`; provide `plan` and let CloudAI generate the JSON.")

        if self.plan is None:
            raise ValueError("NixlEP requires `plan` so CloudAI can generate a per-run JSON file.")

        if isinstance(self.plan, list):
            if len(self.plan) != 1:
                raise ValueError("plan must be a single serialized plan string.")
            self.plan = self.plan[0]

        if isinstance(self.plan, str):
            self.plan = self.plan.strip()
            if not self.plan:
                raise ValueError("plan must not be empty.")
            self.parse_plan()

        return self

    def parse_plan(self) -> list[list[int]]:
        if self.plan is None:
            raise ValueError("parse_plan() requires cmd_args.plan to be set.")
        if not isinstance(self.plan, str):
            raise ValueError("parse_plan() requires cmd_args.plan to be a serialized string.")

        try:
            parsed = json.loads(self.plan)
        except json.JSONDecodeError as exc:
            raise ValueError(f"plan must be valid JSON: {exc}") from exc

        if not isinstance(parsed, list) or not parsed:
            raise ValueError("plan must decode to a non-empty list of phases.")

        for phase in parsed:
            if not isinstance(phase, list) or not phase:
                raise ValueError("Each plan phase must be a non-empty list of ranks.")
            if any(not isinstance(rank, int) for rank in phase):
                raise ValueError("Each plan rank must be an integer.")

        return parsed


class NixlEPTestDefinition(TestDefinition):
    """Test definition for the NIXL Elastic EP benchmark."""

    container_runtime_root: ClassVar[str] = "/workspace/nixl"
    container_plugin_dir: ClassVar[str] = "/usr/local/nixl/lib/x86_64-linux-gnu/plugins"
    container_python_path: ClassVar[str] = "/usr/local/nixl/lib/python3/dist-packages"
    container_library_dir: ClassVar[str] = "/usr/local/nixl/lib"
    launcher_failure_patterns: ClassVar[tuple[tuple[str, str], ...]] = (
        ("python3: can't open file", "The benchmark entrypoint could not be opened."),
        ("Traceback (most recent call last):", "The benchmark launcher raised a Python traceback."),
        ("Timed out waiting for NIXL EP master services", "The master services never became ready."),
        ("no plan phases were found for rank", "A worker was launched for a rank that never appears in the plan."),
        ("recvValueWithTimeout failed", "A worker lost its TCPStore connection before the benchmark completed."),
        ("timed out after 300000ms", "A worker timed out waiting on the TCPStore."),
        ("Failed to prepare remote memory view", "NIXL EP failed to initialize its UCX remote memory view."),
        ("srun: error:", "Slurm reported an srun failure."),
        ("Exited with exit code", "A Slurm step exited with a non-zero status."),
    )
    cmd_args: NixlEPCmdArgs
    _docker_image: Optional[DockerImage] = None

    @classmethod
    def _normalize_config_repos(cls, repos: list[GitRepo]) -> list[GitRepo]:
        normalized = list(repos)
        for repo in normalized:
            if (repo.mount_as or "").rstrip("/") == cls.container_runtime_root.rstrip("/"):
                raise ValueError(
                    "NixlEP git_repos must not mount to '/workspace/nixl' because that shadows the container's "
                    "prebuilt runtime."
                )
        return normalized

    @model_validator(mode="after")
    def validate_git_repos(self) -> "NixlEPTestDefinition":
        self.git_repos = self._normalize_config_repos(self.git_repos)
        return self

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def container_runtime_root_path(self) -> PurePosixPath:
        return PurePosixPath(self.container_runtime_root)

    def resolve_elastic_script_path(self) -> str:
        path = PurePosixPath(self.cmd_args.elastic_script)
        if path.is_absolute():
            return str(path)
        return str(self.container_runtime_root_path / path)

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]

    def _expected_node_logs(self, tr: TestRun) -> list[Path]:
        return [tr.output_path / f"nixl-ep-node-{node_idx}.log" for node_idx in range(tr.num_nodes)]

    @staticmethod
    def _tail(path: Path, num_lines: int = 40) -> str:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-num_lines:])

    def _scan_log_for_failures(self, path: Path) -> JobStatusResult | None:
        if not path.is_file():
            return None

        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, description in self.launcher_failure_patterns:
            if pattern in content:
                tail = self._tail(path)
                error_message = f"{description} See {path}."
                if tail:
                    error_message += f"\n{tail}"
                return JobStatusResult(is_successful=False, error_message=error_message)

        return None

    def _check_slurm_job_status(self, status_path: Path) -> JobStatusResult | None:
        if not status_path.is_file():
            return None

        try:
            status = toml.loads(status_path.read_text(encoding="utf-8", errors="ignore"))
        except toml.TomlDecodeError as exc:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Failed to parse Slurm job status file {status_path}: {exc}",
            )

        state = status.get("state")
        if state == "COMPLETED":
            return None

        job_exit_code = status.get("exit_code", "unknown")
        for step in reversed(status.get("job_steps", [])):
            if step.get("state") != "COMPLETED":
                step_id = step.get("step_id", "unknown")
                step_name = step.get("name", "unknown")
                step_exit_code = step.get("exit_code", "unknown")
                submit_line = step.get("submit_line", "")
                details = (
                    f"NIXL EP Slurm job did not complete successfully "
                    f"(state={state}, exit_code={job_exit_code}). "
                    f"Last failing step: {step_id} ({step_name}), exit_code={step_exit_code}."
                )
                if submit_line:
                    details += f"\nCommand: {submit_line}"
                return JobStatusResult(is_successful=False, error_message=details)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"NIXL EP Slurm job did not complete successfully "
                f"(state={state}, exit_code={job_exit_code})."
            ),
        )

    def _check_benchmark_output(self, expected_node_logs: list[Path]) -> JobStatusResult | None:
        if any(parse_nixl_ep_bandwidth_samples(path) for path in expected_node_logs):
            return None

        first_log = expected_node_logs[0]
        tail = self._tail(first_log)
        error_message = (
            "NIXL EP completed at the Slurm level, but no benchmark summary lines were found in the node logs. "
            "Expected lines such as '[rank N] Dispatch + combine bandwidth: ...'. "
            f"Checked logs: {', '.join(path.name for path in expected_node_logs)}."
        )
        if tail:
            error_message += f"\n{tail}"

        return JobStatusResult(is_successful=False, error_message=error_message)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        expected_node_logs = self._expected_node_logs(tr)

        for path in [*expected_node_logs, output_path / "stdout.txt", output_path / "stderr.txt"]:
            result = self._scan_log_for_failures(path)
            if result is not None:
                return result

        missing_node_logs = [path.name for path in expected_node_logs if not path.is_file()]
        if missing_node_logs:
            existing_node_logs = sorted(path.name for path in output_path.glob("nixl-ep-node-*.log"))
            existing_logs_str = ", ".join(existing_node_logs) if existing_node_logs else "none"
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Expected NIXL EP node logs not found in {output_path}: {', '.join(missing_node_logs)}. "
                    f"Existing node logs: {existing_logs_str}."
                ),
            )

        status_result = self._check_slurm_job_status(output_path / "slurm-job.toml")
        if status_result is not None:
            return status_result

        benchmark_output_result = self._check_benchmark_output(expected_node_logs)
        if benchmark_output_result is not None:
            return benchmark_output_result

        return JobStatusResult(is_successful=True)
