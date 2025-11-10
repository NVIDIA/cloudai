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
from cloudai.workloads.common.nixl import extract_nixlbench_data


class NIXLBenchCmdArgs(CmdArgs):
    """Command line arguments for a NIXL Bench test."""

    docker_image_url: str
    path_to_benchmark: str
    etcd_path: str = "etcd"
    etcd_endpoints: str = "http://$NIXL_ETCD_ENDPOINTS"


class NIXLBenchTestDefinition(TestDefinition):
    """Test definition for a NIXL Bench test."""

    cmd_args: NIXLBenchCmdArgs
    _nixl_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._nixl_image:
            self._nixl_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._nixl_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(exclude={"docker_image_url", "path_to_benchmark", "cmd_args", "etcd_path"})

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        df = extract_nixlbench_data(tr.output_path / "stdout.txt")
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)
