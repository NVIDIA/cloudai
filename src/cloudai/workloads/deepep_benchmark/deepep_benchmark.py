# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class DeepEPBenchmarkCmdArgs(CmdArgs):
    """DeepEP benchmark command arguments."""

    docker_image_url: Optional[str] = None
    mode: Literal["standard", "low_latency"] = "standard"
    tokens: int = 1024
    num_experts: int = 256
    num_topk: int = 8
    hidden_size: int = 7168
    data_type: Literal["bfloat16", "fp8"] = "bfloat16"
    allow_nvlink_for_low_latency: bool = False
    allow_mnnvl: bool = False
    round_scale: bool = False
    use_ue8m0: bool = False
    num_warmups: int = 20
    num_iterations: int = 50
    shuffle_columns: bool = False
    use_kineto_profiler: bool = False
    num_sms: int = 24
    num_qps_per_rank: int = 12
    config_file_path: str = "/tmp/config.yaml"
    results_dir: str = "/workspace/dp-benchmark/results"


class DeepEPBenchmarkTestDefinition(TestDefinition):
    """Test object for DeepEP MoE benchmark."""

    cmd_args: DeepEPBenchmarkCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            if not self.cmd_args.docker_image_url:
                raise ValueError("docker_image_url is required for DeepEP benchmark")
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
