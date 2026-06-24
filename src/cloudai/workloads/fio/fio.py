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

from pathlib import Path
from typing import Any, Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .report_generation_strategy import extract_fio_data

FioArgValue = bool | int | float | str | list[bool | int | float | str]


class FioCmdArgs(CmdArgs):
    """Command line arguments for fio."""

    fio_binary: str = "fio"
    """fio executable to run. Use an absolute path for patched/custom fio builds."""

    job_file: Optional[str] = None
    """Optional fio job/config file. When set, it is appended after CLI options."""

    args: dict[str, FioArgValue] = Field(default_factory=dict)
    """fio CLI options, without leading ``--``. Underscores are converted to dashes."""

    passthrough_args: list[str] = Field(default_factory=list)
    """Additional raw fio CLI arguments appended after ``args``."""

    docker_image_url: Optional[str] = None
    """Optional Docker image to use for Slurm container execution."""

    num_tasks: Optional[int] = None
    """Optional total Slurm task count. Defaults to ``num_nodes * num_tasks_per_node``."""

    num_tasks_per_node: Optional[int] = 1
    """Optional Slurm task count per node for multi-node fio runs."""

    @model_validator(mode="after")
    def validate_launch_mode(self) -> Self:
        if not self.job_file and not self.args and not self.passthrough_args:
            raise ValueError("fio requires at least one of job_file, args, or passthrough_args.")
        return self


class FioTestDefinition(TestDefinition):
    """Test definition for fio."""

    cmd_args: FioCmdArgs
    dse_excluded_args: list[str] = Field(default_factory=lambda: ["cmd_args.passthrough_args"])
    _fio_image: DockerImage | None = None

    @model_validator(mode="after")
    def exclude_passthrough_args_from_dse(self) -> Self:
        if "cmd_args.passthrough_args" not in self.dse_excluded_args:
            self.dse_excluded_args.append("cmd_args.passthrough_args")
        return self

    @property
    def docker_image(self) -> DockerImage | None:
        if not self.cmd_args.docker_image_url:
            return None
        if not self._fio_image:
            self._fio_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._fio_image

    @property
    def installables(self) -> list[Installable]:
        image = self.docker_image
        return [*([image] if image else []), *self.git_repos]

    @property
    def cmd_args_dict(self) -> dict[str, Any]:
        return self.cmd_args.model_dump(
            exclude={
                "fio_binary",
                "job_file",
                "docker_image_url",
                "num_tasks",
                "num_tasks_per_node",
                "passthrough_args",
            },
            exclude_none=True,
        )

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stdout_path = tr.output_path / "stdout.txt"
        stderr_path = tr.output_path / "stderr.txt"
        if not stdout_path.is_file():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"stdout.txt file not found in {tr.output_path}. "
                    f"Check whether fio started successfully and inspect {stderr_path}."
                ),
            )

        if not stdout_path.read_text().strip():
            return JobStatusResult(
                is_successful=False,
                error_message=f"stdout.txt is empty in {tr.output_path}. Check {stderr_path} for fio errors.",
            )

        if not extract_fio_data(Path(stdout_path)):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"fio summary metrics were not found in {stdout_path}. "
                    "Expected fio stdout to include IOPS, BW, and latency lines."
                ),
            )

        return JobStatusResult(is_successful=True)
