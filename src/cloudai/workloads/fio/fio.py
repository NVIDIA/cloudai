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

import pathlib
from typing import Any

import pydantic

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .report_generation_strategy import extract_fio_data


class FioCmdArgs(CmdArgs):
    """Command line arguments for fio."""

    fio_binary: str = "fio"
    """fio executable to run. Use an absolute path for patched/custom fio builds."""

    job_file: str | None = None
    """Optional fio job/config file. When set, it is appended after CLI options."""

    args: dict[str, Any] = pydantic.Field(default_factory=dict)
    """fio CLI options, without leading ``--``. Keys are passed to fio verbatim."""

    docker_image_url: str | None = None
    """Optional Docker image to use for Slurm container execution."""

    num_tasks_per_node: int | None = 1
    """Optional Slurm task count per node for multi-node fio runs."""

    metric_operation: str = "all"
    """Operation used for the default metric: read, write, trim, all, or first."""

    metric_name: str = "bw"
    """Metric used for the default metric: bw, iops, or latency."""

    metric_aggregate: str = "sum"
    """Aggregation used for the default metric: sum, mean, min, max, or first."""

    def fio_args(self) -> dict[str, Any]:
        """Return only arguments intended for the fio CLI."""
        return self.args


class FioTestDefinition(TestDefinition):
    """Test definition for fio."""

    cmd_args: FioCmdArgs
    _fio_image: DockerImage | None = None

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

        if not extract_fio_data(pathlib.Path(stdout_path)):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"fio summary metrics were not found in {stdout_path}. "
                    "Expected fio stdout to include IOPS, BW, and latency lines."
                ),
            )

        return JobStatusResult(is_successful=True)
