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

from typing import Literal

from cloudai.core import CmdArgs, DockerImage, Installable, JobStatusResult, TestDefinition, TestRun
from cloudai.workloads.common.nixl import extract_nixlbench_data


class NIXLKVBenchCmdArgs(CmdArgs):
    """Command line arguments for NIXLKVBench."""

    command: Literal["profile"] = "profile"
    etcd_path: str = "etcd"
    wait_etcd_for: int = 60

    docker_image_url: str
    kvbench_script: str = "/workspace/nixl/benchmark/kvbench/main.py"
    python_executable: str = "python"

    model_cfg: str | list[str] | None = None
    """Path to model configuration file used by NIXL KVBench."""

    backend: str | list[str] | None = None


class NIXLKVBenchTestDefinition(TestDefinition):
    """Test definition for NIXLKVBench."""

    _docker_image: DockerImage | None = None
    cmd_args: NIXLKVBenchCmdArgs

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image]

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(
            exclude={
                "kvbench_script",
                "python_executable",
                "etcd_path",
                "wait_etcd_for",
                "docker_image_url",
                "command",
            },
        )

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        df = extract_nixlbench_data(tr.output_path / "stdout.txt")
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)
