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

from typing import Literal, Optional, Union

from cloudai.core import DockerImage, Installable
from cloudai.models.workload import CmdArgs, TestDefinition


class NCCLCmdArgs(CmdArgs):
    """NCCL test command arguments."""

    docker_image_url: str = "nvcr.io/nvidia/pytorch:24.02-py3"
    subtest_name: Union[
        Literal[
            "all_reduce_perf_mpi",
            "all_gather_perf_mpi",
            "alltoall_perf_mpi",
            "broadcast_perf_mpi",
            "gather_perf_mpi",
            "hypercube_perf_mpi",
            "reduce_perf_mpi",
            "reduce_scatter_perf_mpi",
            "scatter_perf_mpi",
            "sendrecv_perf_mpi",
            "bisection_perf_mpi",
            # K8s tests
            "all_reduce_perf",
            "all_gather_perf",
            "alltoall_perf",
            "broadcast_perf",
            "gather_perf",
            "hypercube_perf",
            "reduce_perf",
            "reduce_scatter_perf",
            "scatter_perf",
            "sendrecv_perf",
            "bisection_perf",
        ],
        list[
            Literal[
                "all_reduce_perf_mpi",
                "all_gather_perf_mpi",
                "alltoall_perf_mpi",
                "broadcast_perf_mpi",
                "gather_perf_mpi",
                "hypercube_perf_mpi",
                "reduce_perf_mpi",
                "reduce_scatter_perf_mpi",
                "scatter_perf_mpi",
                "sendrecv_perf_mpi",
                "bisection_perf_mpi",
                # K8s tests
                "all_reduce_perf",
                "all_gather_perf",
                "alltoall_perf",
                "broadcast_perf",
                "gather_perf",
                "hypercube_perf",
                "reduce_perf",
                "reduce_scatter_perf",
                "scatter_perf",
                "sendrecv_perf",
                "bisection_perf",
            ]
        ],
    ] = "all_reduce_perf_mpi"
    nthreads: Union[int, list[int]] = 1
    ngpus: Union[int, list[int]] = 1
    minbytes: Union[str, list[str]] = "32M"
    maxbytes: Union[str, list[str]] = "32M"
    stepbytes: Union[str, list[str]] = "1M"
    op: Union[
        Literal["sum", "prod", "min", "max", "avg", "all"], list[Literal["sum", "prod", "min", "max", "avg", "all"]]
    ] = "sum"
    datatype: Union[Literal["uint8", "float"], list[Literal["uint8", "float"]]] = "float"
    root: Union[int, list[int]] = 0
    iters: Union[int, list[int]] = 20
    warmup_iters: Union[int, list[int]] = 5
    agg_iters: Union[int, list[int]] = 1
    average: Union[int, list[int]] = 1
    parallel_init: Union[int, list[int]] = 0
    check: Union[int, list[int]] = 1
    blocking: Union[int, list[int]] = 0
    cudagraph: Union[int, list[int]] = 0


class NCCLTestDefinition(TestDefinition):
    """Test object for NCCL."""

    cmd_args: NCCLCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.predictor] if self.predictor else [self.docker_image]
