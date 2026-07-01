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

from typing import Literal, Optional

from pydantic import Field

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class MoEBenchmarkCmdArgs(CmdArgs):
    """Command arguments for the custom MoE benchmark that compares EP/alltoallv backends."""

    docker_image_url: str
    benchmark_root: str = "/workspace/DeepEP/benchmark"
    mode: Literal["standard", "low_latency"] = "standard"
    deepep_versions: list[str] = Field(default_factory=lambda: ["legacy", "elastic"])
    tokens: int = 1024
    num_experts: int = 256
    num_topk: int = 8
    hidden_size: int = 7168
    data_type: Literal["bfloat16", "fp8"] = "bfloat16"
    allow_nvlink_for_low_latency: bool = False
    allow_mnnvl: bool = False
    round_scale: bool = False
    use_ue8m0: bool = False
    benchmark_combine: bool = True
    num_warmups: int = 20
    num_iterations: int = 50
    shuffle_columns: bool = False
    use_kineto_profiler: bool = False
    enable_tuning: bool = False
    num_sms: int = 24
    num_qps_per_rank: int = 12

    v2_num_sms: int = 12
    v2_num_qps: int = 0
    v2_prefer_overlap_with_compute: bool = False
    config_file_path: str = "/tmp/config.yaml"
    results_dir: str = "/workspace/DeepEP/results"


class MoEBenchmarkTestDefinition(TestDefinition):
    """Test object for the custom MoE benchmark."""

    cmd_args: MoEBenchmarkCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            if not self.cmd_args.docker_image_url:
                raise ValueError("docker_image_url is required for MoE benchmark")
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

    @property
    def cmd_args_dict(self) -> dict:
        """Return command arguments as dict, excluding CloudAI/container-only fields."""
        return self.cmd_args.model_dump(
            exclude={
                "docker_image_url",
                "benchmark_root",
                "mode",
                "deepep_versions",
                "num_sms",
                "num_qps_per_rank",
                "config_file_path",
                "results_dir",
            }
        )
