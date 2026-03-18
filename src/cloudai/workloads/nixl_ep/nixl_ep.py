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

from pathlib import Path, PurePosixPath
from typing import ClassVar, Optional

import toml
from pydantic import AliasChoices, Field, field_validator, model_validator

from cloudai.core import DockerImage, GitRepo, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class NixlEPCmdArgs(CmdArgs):
    """Command line arguments for the NIXL Elastic EP benchmark."""

    docker_image_url: str = Field(description="URL of the Docker image that contains the NIXL EP benchmark.")
    elastic_script: str = Field(
        default="examples/device/ep/tests/elastic/elastic.py",
        description=(
            "Path to the benchmark entrypoint, relative to the benchmark repo mount "
            "or absolute in the container."
        ),
    )
    python_executable: str = Field(default="python3", description="Python executable to use inside the container.")
    input_json: str = Field(
        validation_alias=AliasChoices("input_json", "plan"),
        description="Path to the phase plan JSON, relative to the benchmark repo mount or absolute in the container.",
    )
    num_processes_per_node: int | list[int] = Field(
        description="Number of local worker processes to spawn on each allocated node.",
    )
    num_tokens: int = Field(default=128, ge=1, description="Tokens per dispatch.")
    num_experts_per_rank: int = Field(default=2, ge=1, description="Experts per rank.")
    hidden_dim: int = Field(default=7168, ge=1, description="Hidden dimension.")
    num_topk: int = Field(default=8, ge=1, description="Top-K routing value.")
    disable_ll_nvlink: bool = Field(default=False, description="Disable low-latency NVLink kernels.")
    kineto: bool = Field(default=False, description="Enable Kineto profiling.")
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


class NixlEPTestDefinition(TestDefinition):
    """Test definition for the NIXL Elastic EP benchmark."""

    benchmark_repo_mount: ClassVar[str] = "/workspace/nixl"
    launcher_failure_patterns: ClassVar[tuple[tuple[str, str], ...]] = (
        ("python3: can't open file", "The benchmark entrypoint could not be opened."),
        ("Traceback (most recent call last):", "The benchmark launcher raised a Python traceback."),
        ("Timed out waiting for NIXL EP master services", "The master services never became ready."),
        ("srun: error:", "Slurm reported an srun failure."),
        ("Exited with exit code", "A Slurm step exited with a non-zero status."),
    )
    cmd_args: NixlEPCmdArgs
    _docker_image: Optional[DockerImage] = None

    @staticmethod
    def _is_nixl_repo(repo: GitRepo) -> bool:
        return "nixl" in repo.url.lower()

    @classmethod
    def _normalize_benchmark_repo(cls, repos: list[GitRepo]) -> list[GitRepo]:
        if not repos:
            raise ValueError(
                "NixlEP requires the benchmark repository via `[[git_repos]]` so the launcher and input JSON are "
                "available inside the container."
            )

        normalized = list(repos)
        benchmark_idx = next(
            (
                idx
                for idx, repo in enumerate(normalized)
                if (repo.mount_as or "").rstrip("/") == cls.benchmark_repo_mount.rstrip("/")
            ),
            None,
        )
        if benchmark_idx is None and len(normalized) == 1:
            benchmark_idx = 0
        if benchmark_idx is None:
            benchmark_idx = next((idx for idx, repo in enumerate(normalized) if cls._is_nixl_repo(repo)), None)
        if benchmark_idx is None:
            raise ValueError(
                "NixlEP requires one benchmark repo in `[[git_repos]]`. When multiple repos are present, set "
                "mount_as='/workspace/nixl' on the NIXL benchmark repo."
            )

        repo = normalized[benchmark_idx]
        if (repo.mount_as or "").rstrip("/") != cls.benchmark_repo_mount.rstrip("/"):
            normalized[benchmark_idx] = repo.model_copy(update={"mount_as": cls.benchmark_repo_mount})

        return normalized

    @model_validator(mode="after")
    def validate_git_repos(self) -> "NixlEPTestDefinition":
        self.git_repos = self._normalize_benchmark_repo(self.git_repos)
        return self

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def benchmark_repo(self) -> GitRepo:
        for repo in self.git_repos:
            if (repo.mount_as or "").rstrip("/") == self.benchmark_repo_mount.rstrip("/"):
                return repo
        raise ValueError("NixlEP benchmark repo was not normalized to the expected mount path.")

    @property
    def benchmark_repo_root(self) -> PurePosixPath:
        return PurePosixPath(self.benchmark_repo.container_mount)

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

        return JobStatusResult(is_successful=True)
