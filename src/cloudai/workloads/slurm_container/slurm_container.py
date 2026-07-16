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

import toml
from pydantic import Field, ValidationError

from cloudai.core import DockerImage, File, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmJobMetadata


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
        """Grade the run from the real container exit code recorded by Slurm."""
        slurm_job_path = tr.output_path / "slurm-job.toml"
        if not slurm_job_path.is_file():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"slurm-job.toml file not found in the specified output directory {tr.output_path}. "
                    "This file is required to determine the container exit status."
                ),
            )

        try:
            with slurm_job_path.open("r", encoding="utf-8") as file:
                metadata = SlurmJobMetadata.model_validate(toml.load(file))
        except (OSError, toml.TomlDecodeError, ValidationError) as err:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Failed to read Slurm job metadata from {slurm_job_path} "
                    f"(malformed or partially written slurm-job.toml?): {err}"
                ),
            )

        if metadata.exit_code == "0" or metadata.exit_code.startswith("0:"):
            return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"Container command exited with a non-zero exit code for {tr.output_path}: "
                f"state={metadata.state}, exit_code={metadata.exit_code}."
            ),
        )
