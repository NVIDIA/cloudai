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

import os
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.systems import SlurmSystem
from cloudai.workloads.deepseek_r1 import (
    DeepSeekR1CmdArgs,
    DeepSeekR1SlurmCommandGenStrategy,
    DeepSeekR1TestDefinition,
)


@pytest.fixture
def cmd_gen_strategy(slurm_system: SlurmSystem) -> DeepSeekR1SlurmCommandGenStrategy:
    return DeepSeekR1SlurmCommandGenStrategy(slurm_system, {})


@pytest.mark.parametrize(
    ("num_nodes", "nodes"),
    [
        (1, ["nodeA"]),
        (3, ["n1", "n2", "n3"]),
    ],
)
def test_parse_slurm_args_includes_image_path(
    cmd_gen_strategy: DeepSeekR1SlurmCommandGenStrategy,
    num_nodes: int,
    nodes: List[str],
) -> None:
    cmd_args = DeepSeekR1CmdArgs()
    tdef = DeepSeekR1TestDefinition(
        name="dsr1",
        description="desc",
        test_template_name="tt",
        cmd_args=cmd_args,
        extra_env_vars={},
    )
    test = Test(test_definition=tdef, test_template=Mock())
    tr = TestRun(name="run", test=test, nodes=nodes, num_nodes=num_nodes)
    base_args = cmd_gen_strategy._parse_slurm_args("dsr1", {}, {}, tr)
    assert base_args["image_path"] == tdef.docker_image.installed_path


def test_append_sbatch_directives_no_credentials(
    tmp_path: Path,
    cmd_gen_strategy: DeepSeekR1SlurmCommandGenStrategy,
) -> None:
    os.environ["HOME"] = str(tmp_path / "home")
    batch_lines: List[str] = []
    args = {"num_nodes": 2, "node_list_str": ""}
    cmd_gen_strategy._append_sbatch_directives(batch_lines, args, output_path=Path("/out"))
    assert "export HEAD_NODE=$SLURM_JOB_MASTER_NODE" in batch_lines
    assert "export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE" in batch_lines
    assert "export NIM_NUM_COMPUTE_NODES=2" in batch_lines
    assert any("WARNING: Failed to load NGC API key" in line for line in batch_lines)


def test_append_sbatch_directives_with_credentials(
    tmp_path: Path,
    cmd_gen_strategy: DeepSeekR1SlurmCommandGenStrategy,
) -> None:
    cred_dir = tmp_path / "home" / ".config" / "enroot"
    cred_dir.mkdir(parents=True, exist_ok=True)
    (cred_dir / ".credentials").write_text("machine nvcr.io login user password SECRET")
    os.environ["HOME"] = str(tmp_path / "home")
    batch_lines: List[str] = []
    args = {"num_nodes": 5, "node_list_str": ""}
    cmd_gen_strategy._append_sbatch_directives(batch_lines, args, output_path=Path("/out"))
    assert "export NGC_API_KEY=SECRET" in batch_lines


def test_generate_test_command_returns_startup_script(
    cmd_gen_strategy: DeepSeekR1SlurmCommandGenStrategy,
) -> None:
    result = cmd_gen_strategy.generate_test_command({}, {}, Mock())
    assert result == ["/opt/nim/start_server.sh"]
