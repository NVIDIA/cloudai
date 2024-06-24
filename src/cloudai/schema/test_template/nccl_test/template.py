# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from cloudai import TestTemplate


class NcclTest(TestTemplate):
    """
    Test template for NCCL tests.

    Attributes
        SUPPORTED_SUBTESTS (List[str]): List of supported subtests for NCCL,
            including all_reduce_perf_mpi, all_gather_perf_mpi, and others.
    """

    SUPPORTED_SUBTESTS = [
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
    ]
