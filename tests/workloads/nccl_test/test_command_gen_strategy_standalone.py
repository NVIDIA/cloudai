# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shlex
from pathlib import Path

import pytest
import toml

from cloudai.core import TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.nccl_test import NCCLTestDefinition, NcclTestStandaloneMPICommandGenStrategy


def test_standalone_mpi(standalone_system: StandaloneSystem, tmp_path: Path) -> None:
    test = NCCLTestDefinition.model_validate(
        {
            "name": "nccl",
            "description": "MPI",
            "test_template_name": "NcclTest",
            "standalone_mpi": {"binary_dir": "/shared/tests", "ranks_per_node": 4},
            "cmd_args": {"subtest_name": "all_reduce_perf", "minbytes": "8G"},
            "extra_env_vars": {"NCCL_ALGO": "Ring"},
            "extra_cmd_args": {"--local_register": "2"},
        }
    )
    assert not test.installables
    tr = TestRun(name="mpi", test=test, nodes=["node01", "node02"], num_nodes=2, output_path=tmp_path)
    strategy = NcclTestStandaloneMPICommandGenStrategy(standalone_system, tr)
    command = strategy.gen_exec_command()
    details = toml.load(tmp_path / "test-run.toml")
    assert command == details["full_cmd"]
    args = shlex.split(details["test_cmd"])
    assert args[args.index("--host") + 1] == "node01:4,node02:4"
    assert args[args.index("-np") + 1] == "8"
    assert "NCCL_ALGO=Ring" in args
    assert args[args.index("NCCL_ALGO") - 1] == "-x"
    assert "/shared/tests/all_reduce_perf" in args
    assert args[args.index("--minbytes") + 1] == "8G"
    assert args[-2:] == ["--local_register", "2"]
    tr.nodes = ["node01"]
    with pytest.raises(ValueError, match="distinct MPI hostnames"):
        strategy.gen_exec_command()
