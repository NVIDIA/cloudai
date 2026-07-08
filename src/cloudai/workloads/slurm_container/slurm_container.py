# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from pydantic import Field

from cloudai.core import DockerImage, File, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

EXIT_CODE_FILE_NAME = "exit_code.txt"


class SlurmContainerCmdArgs(CmdArgs):
    """Command line arguments for a generic Slurm container test."""

    docker_image_url: str
    cmd: str


class SlurmContainerTestDefinition(TestDefinition):
    """Test definition for a generic Slurm container test."""

    cmd_args: SlurmContainerCmdArgs
    extra_srun_args: list[str] = Field(default_factory=list)
    scripts: list[File] = Field(default_factory=list)
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos, *self.scripts]

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        """Grade the run from the container command exit code."""
        exit_code_path = tr.output_path / EXIT_CODE_FILE_NAME
        if not exit_code_path.is_file():
            return JobStatusResult(
                is_successful=False,
                error_message=f"Exit code file {exit_code_path} not found.",
            )

        try:
            exit_code_text = exit_code_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as err:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Could not read exit code file {exit_code_path}: {err}.",
            )

        try:
            exit_code = int(exit_code_text)
        except ValueError:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Could not parse exit code from {exit_code_path}: {exit_code_text!r}.",
            )

        if exit_code != 0:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Container command exited with code {exit_code}.",
            )

        return JobStatusResult(is_successful=True)
