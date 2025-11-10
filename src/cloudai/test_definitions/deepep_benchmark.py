# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai import CmdArgs, TestDefinition
from cloudai.installer.installables import DockerImage, Installable


class DeepEPBenchmarkCmdArgs(CmdArgs):
    """DeepEP benchmark command arguments."""

    docker_image_url: str = "gitlab-master.nvidia.com/ybenabou/warehouse/deepep:dp-benchmark"
    mode: Literal["standard", "low_latency"] = "standard"
    
    # Basic parameters
    tokens: int = 1024
    num_experts: int = 256
    num_topk: int = 8
    hidden_size: int = 7168
    
    # Data type
    data_type: Literal["bfloat16", "fp8"] = "bfloat16"
    
    # Low-latency mode settings
    allow_nvlink_for_low_latency: bool = False
    allow_mnnvl: bool = False
    
    # FP8 settings
    round_scale: bool = False
    use_ue8m0: bool = False
    
    # Benchmark settings
    num_warmups: int = 20
    num_iterations: int = 50
    shuffle_columns: bool = False
    use_kineto_profiler: bool = False
    
    # Environment variables for standard mode
    num_sms: int = 24
    num_qps_per_rank: int = 12
    
    # Config file path (will be mounted inside container)
    config_file_path: str = "/tmp/config.yaml"
    
    # Results directory (will be mounted inside container)
    results_dir: str = "/workspace/dp-benchmark/results"


class DeepEPBenchmarkTestDefinition(TestDefinition):
    """Test object for DeepEP MoE benchmark."""

    cmd_args: DeepEPBenchmarkCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

