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
import re
from pathlib import Path, PurePosixPath
from typing import ClassVar, Optional

from pydantic import Field, field_validator

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .log_parsing import parse_nixl_ep_bandwidth_samples

GENERATED_PLAN_FILE_NAME = "nixl-ep-plan.json"


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
    plan: str | list[str] = Field(
        description=(
            "Serialized phase plan to write into a per-run JSON file. "
            'Use a single string such as "[[0, 1], [0, 1, 2, 3]]" for a single run, '
            "or a list of such strings to enable DSE mode (one run per plan)."
        ),
    )
    num_processes_per_node: int | list[int] = Field(
        description="Number of local worker processes to spawn on each allocated node.",
    )
    service_startup_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Seconds to wait for the master node's TCPStore to accept connections.",
    )
    store_port: int = Field(default=9999, ge=1, le=65535, description="TCPStore port used by the benchmark.")

    @field_validator("num_processes_per_node", mode="after")
    @classmethod
    def validate_num_processes_per_node(cls, value: int | list[int]) -> int | list[int]:
        values = value if isinstance(value, list) else [value]
        if any(item < 1 for item in values):
            raise ValueError("num_processes_per_node must contain only positive integers")
        return value

    @field_validator("plan", mode="after")
    @classmethod
    def validate_plan(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            if not value:
                raise ValueError("plan list must not be empty.")
            for item in value:
                stripped = item.strip()
                if not stripped:
                    raise ValueError("plan list must not contain empty strings.")
                cls._parse_plan(stripped)
            return value
        value = value.strip()
        if not value:
            raise ValueError("plan must not be empty.")
        cls._parse_plan(value)
        return value

    @staticmethod
    def _parse_plan(plan: str) -> list[list[int]]:
        try:
            parsed = json.loads(plan)
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

    def parse_plan(self) -> list[list[int]]:
        if not isinstance(self.plan, str):
            raise ValueError("parse_plan() requires cmd_args.plan to be a serialized string.")
        return self._parse_plan(self.plan)


class NixlEPTestDefinition(TestDefinition):
    """Test definition for the NIXL Elastic EP benchmark."""

    container_runtime_root: ClassVar[str] = "/workspace/nixl"
    cmd_args: NixlEPCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]

    @property
    def container_runtime_root_path(self) -> PurePosixPath:
        return PurePosixPath(self.container_runtime_root)

    @staticmethod
    def _tail(path: Path, num_lines: int = 40) -> str:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-num_lines:])

    @staticmethod
    def _primary_launch_exit_error_message(content: str) -> str | None:
        match = re.search(r"Primary NIXL EP launch exited before phase (\d+) completed", content)
        if match is None:
            return None

        phase = int(match.group(1))
        if phase == 0:
            return (
                "The initial NIXL EP launch exited before phase 0 completed, so later stage launches never "
                "started and some node logs may be absent."
            )

        return f"The primary NIXL EP launch exited before phase {phase} completed."

    def _scan_log_for_failures(self, path: Path) -> JobStatusResult | None:
        if not path.is_file():
            return None

        launcher_failure_patterns = (
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
        content = path.read_text(encoding="utf-8", errors="ignore")
        primary_launch_error = self._primary_launch_exit_error_message(content)
        if primary_launch_error is not None:
            tail = self._tail(path)
            error_message = f"{primary_launch_error} See {path}."
            if tail:
                error_message += f"\n{tail}"
            return JobStatusResult(is_successful=False, error_message=error_message)

        for pattern, description in launcher_failure_patterns:
            if pattern in content:
                tail = self._tail(path)
                error_message = f"{description} See {path}."
                if tail:
                    error_message += f"\n{tail}"
                return JobStatusResult(is_successful=False, error_message=error_message)

        return None

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
        expected_node_logs = [tr.output_path / f"nixl-ep-node-{node_idx}.log" for node_idx in range(tr.nnodes)]

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

        benchmark_output_result = self._check_benchmark_output(expected_node_logs)
        if benchmark_output_result is not None:
            return benchmark_output_result

        return JobStatusResult(is_successful=True)
