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

import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class NIXLBenchCmdArgs(CmdArgs):
    """Command line arguments for a NIXL Bench test."""

    docker_image_url: str
    path_to_benchmark: str
    etcd_path: str = "etcd"
    etcd_endpoints: str = "http://$NIXL_ETCD_ENDPOINTS"


class NIXLBenchTestDefinition(TestDefinition):
    """Test definition for a NIXL Bench test."""

    cmd_args: NIXLBenchCmdArgs
    _nixl_image: Optional[DockerImage] = None

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
        df = extract_nixl_data(tr.output_path / "stdout.txt")
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)


@cache
def extract_nixl_data(stdout_file: Path) -> pd.DataFrame:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return lazy.pd.DataFrame()

    header_present, data = False, []
    for line in stdout_file.read_text().splitlines():
        if not header_present and (
            "Block Size (B)      Batch Size     " in line and "Avg Lat. (us)" in line and "B/W (GB/Sec)" in line
        ):
            header_present = True
            continue
        parts = line.split()
        if header_present and (len(parts) == 6 or len(parts) == 10):
            if len(parts) == 6:
                data.append([parts[0], parts[1], parts[2], parts[-1]])
            else:
                data.append([parts[0], parts[1], parts[3], parts[2]])

    df = lazy.pd.DataFrame(data, columns=["block_size", "batch_size", "avg_lat", "bw_gb_sec"])
    df["block_size"] = df["block_size"].astype(int)
    df["batch_size"] = df["batch_size"].astype(int)
    df["avg_lat"] = df["avg_lat"].astype(float)
    df["bw_gb_sec"] = df["bw_gb_sec"].astype(float)

    return df
