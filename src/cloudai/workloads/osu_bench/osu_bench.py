# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class OSUBenchCmdArgs(CmdArgs):
    """Command line arguments for a OSU Benchmark test."""

    docker_image_url: str
    path_to_benchmark: str


class OSUBenchTestDefinition(TestDefinition):
    """Test definition for a OSU Benchmark test."""

    cmd_args: OSUBenchCmdArgs
    _osu_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._osu_image:
            self._osu_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._osu_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(exclude={"docker_image_url", "path_to_benchmark"})

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)
