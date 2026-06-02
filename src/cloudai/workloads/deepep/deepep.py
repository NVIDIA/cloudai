# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import PurePosixPath
from typing import ClassVar, Literal, Optional

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class DeepEPCmdArgs(CmdArgs):
    """Command arguments for the official DeepEP test scripts."""

    docker_image_url: str
    subtest_name: Literal["test_internode", "test_intranode", "test_low_latency", "test_ep"] = "test_internode"
    deep_ep_root: str = "/workspace/DeepEP"
    legacy_tests_root: Optional[str] = None
    elastic_tests_root: Optional[str] = None
    python_executable: str = "python"

    num_processes: int = 8
    num_tokens: int = 4096
    hidden: int = 7168
    num_topk: int = 8
    num_experts: int = 256

    # V1 legacy internode/intranode/low-latency flags.
    num_topk_groups: Optional[int] = None
    allow_mnnvl: bool = False
    test_ll_compatibility: bool = False
    pressure_test_mode: int = 0
    pressure_test: bool = False
    shrink_test: bool = False
    disable_nvlink: bool = False
    use_logfmt: bool = False
    shuffle_expert_columns: bool = False
    shuffle_seed: int = 1

    # V2 elastic/test_ep flags.
    num_sms: int = 0
    num_qps: int = 0
    num_allocated_qps: int = 0
    num_gpu_timeout_secs: int = 100
    num_cpu_timeout_secs: int = 100
    sl_idx: int = 0
    do_cpu_sync: int = 1
    allow_hybrid_mode: int = 1
    allow_multiple_reduction: int = 1
    prefer_overlap_with_compute: int = 0
    deterministic: bool = False
    seed: int = 0
    skip_check: bool = False
    skip_perf_test: bool = False
    do_pressure_test: bool = False
    reuse_elastic_buffer: bool = False
    test_first_only: bool = False
    unbalanced_ratio: float = 1.0
    precise_unbalanced_ratio: bool = False
    masked_ratio: float = 0.0
    dump_profile_traces: str = ""
    ignore_local_traffic: bool = False


class DeepEPTestDefinition(TestDefinition):
    """Test object for official DeepEP v1/v2 test scripts."""

    container_runtime_root: ClassVar[str] = "/workspace/DeepEP"
    cmd_args: DeepEPCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            if not self.cmd_args.docker_image_url:
                raise ValueError("docker_image_url is required for DeepEP tests")
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

    @property
    def container_runtime_root_path(self) -> PurePosixPath:
        return PurePosixPath(self.cmd_args.deep_ep_root or self.container_runtime_root)
