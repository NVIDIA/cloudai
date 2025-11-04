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

from pathlib import Path

import pytest

from cloudai import Test, TestRun
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


@pytest.fixture
def nccl_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NCCLTestDefinition(
            name="nccl",
            description="desc",
            test_template_name="t",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        )
    )
    tr = TestRun(name="nccl_test", test=test, num_nodes=2, nodes=[], output_path=tmp_path)

    stdout_content = """# Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] NVIDIA H100
# Rank  1 Group  0 Pid 1001 on node1 device  1 [0xbb] NVIDIA H100
# Rank  2 Group  0 Pid 1002 on node1 device  2 [0xcc] NVIDIA H100
# Rank  3 Group  0 Pid 1003 on node1 device  3 [0xdd] NVIDIA H100
# Rank  4 Group  0 Pid 1004 on node1 device  4 [0xee] NVIDIA H100
# Rank  5 Group  0 Pid 1005 on node1 device  5 [0xff] NVIDIA H100
# Rank  6 Group  0 Pid 1006 on node1 device  6 [0x11] NVIDIA H100
# Rank  7 Group  0 Pid 1007 on node1 device  7 [0x22] NVIDIA H100
# Rank  8 Group  0 Pid 2000 on node2 device  0 [0xaa] NVIDIA H100
# Rank  9 Group  0 Pid 2001 on node2 device  1 [0xbb] NVIDIA H100
# Rank 10 Group  0 Pid 2002 on node2 device  2 [0xcc] NVIDIA H100
# Rank 11 Group  0 Pid 2003 on node2 device  3 [0xdd] NVIDIA H100
# Rank 12 Group  0 Pid 2004 on node2 device  4 [0xee] NVIDIA H100
# Rank 13 Group  0 Pid 2005 on node2 device  5 [0xff] NVIDIA H100
# Rank 14 Group  0 Pid 2006 on node2 device  6 [0x11] NVIDIA H100
# Rank 15 Group  0 Pid 2007 on node2 device  7 [0x22] NVIDIA H100
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1000000       1000000     float     sum      -1    1.11   10.10   20.20      0    1.12   10.11   20.21      0
     2000000       2000000     float     sum      -1    2.22   20.20   30.30      0    2.23   20.21   30.31      0
     12000000      12000000     float     sum      -1   13.13  120.30  130.40      0   13.14  120.31  130.41      0
# Avg bus bandwidth    : 111.111
"""
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr
