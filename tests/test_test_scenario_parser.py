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

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest
from cloudai._core.test import Test
from cloudai._core.test_scenario_parser import TestScenarioParser
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    slurm_system = SlurmSystem(
        name="TestSystem",
        install_path=str(tmp_path / "install"),
        output_path=str(tmp_path / "output"),
        default_partition="main",
        partitions={
            "main": [
                SlurmNode(name="node1", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node2", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node3", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node4", partition="main", state=SlurmNodeState.IDLE),
            ]
        },
    )
    Path(slurm_system.install_path).mkdir()
    Path(slurm_system.output_path).mkdir()
    return slurm_system


@pytest.fixture
def test_mapping() -> Dict[str, Test]:
    mock_test_template = MagicMock()
    return {
        "nccl_test_all_reduce": Test(
            name="nccl_test_all_reduce",
            description="",
            test_template=mock_test_template,
            env_vars={},
            cmd_args={},
            extra_env_vars={},
            extra_cmd_args="",
        ),
        "nccl_test_all_gather": Test(
            name="nccl_test_all_gather",
            description="",
            test_template=mock_test_template,
            env_vars={},
            cmd_args={},
            extra_env_vars={},
            extra_cmd_args="",
        ),
        "nccl_test_reduce_scatter": Test(
            name="nccl_test_reduce_scatter",
            description="",
            test_template=mock_test_template,
            env_vars={},
            cmd_args={},
            extra_env_vars={},
            extra_cmd_args="",
        ),
        "nccl_test_alltoall": Test(
            name="nccl_test_alltoall",
            description="",
            test_template=mock_test_template,
            env_vars={},
            cmd_args={},
            extra_env_vars={},
            extra_cmd_args="",
        ),
        "nccl_test_bisection": Test(
            name="nccl_test_bisection",
            description="",
            test_template=mock_test_template,
            env_vars={},
            cmd_args={},
            extra_env_vars={},
            extra_cmd_args="",
        ),
    }


@pytest.fixture
def test_scenario_parser(
    tmp_path: Path, slurm_system: SlurmSystem, test_mapping: Dict[str, Test]
) -> TestScenarioParser:
    test_file_path = tmp_path / "test_scenario.toml"
    test_scenario_data = """
    name = "nccl-test"

    [nccl_test_all_reduce_1]
      name = "nccl_test_all_reduce"
      num_nodes = "2"
      time_limit = "00:20:00"

    [nccl_test_all_gather_1]
      name = "nccl_test_all_gather"
      num_nodes = "2"
      time_limit = "00:20:00"
      [nccl_test_all_gather_1.dependencies]
        start_post_comp = { name = "nccl_test_all_reduce_1", time = 0 }

    [nccl_test_reduce_scatter_1]
      name = "nccl_test_reduce_scatter"
      num_nodes = "2"
      time_limit = "00:20:00"
      [nccl_test_reduce_scatter_1.dependencies]
        start_post_comp = { name = "nccl_test_all_gather_1", time = 0 }

    [nccl_test_alltoall_1]
      name = "nccl_test_alltoall"
      num_nodes = "2"
      time_limit = "00:20:00"
      [nccl_test_alltoall_1.dependencies]
        start_post_comp = { name = "nccl_test_reduce_scatter_1", time = 0 }

    [nccl_test_bisection_1]
      name = "nccl_test_bisection"
      num_nodes = "2"
      time_limit = "00:20:00"
      [nccl_test_bisection_1.dependencies]
        start_post_comp = { name = "nccl_test_alltoall_1", time = 0 }
    """
    test_file_path.write_text(test_scenario_data)
    return TestScenarioParser(file_path=str(test_file_path), system=slurm_system, test_mapping=test_mapping)


def test_parse_data(test_scenario_parser: TestScenarioParser):
    test_scenario = test_scenario_parser.parse()

    assert test_scenario.name == "nccl-test"
    assert len(test_scenario.tests) == 5

    test_names = [test.name for test in test_scenario.tests]
    assert "nccl_test_all_reduce" in test_names
    assert "nccl_test_all_gather" in test_names
    assert "nccl_test_reduce_scatter" in test_names
    assert "nccl_test_alltoall" in test_names
    assert "nccl_test_bisection" in test_names

    dependencies = {test.name: test.dependencies for test in test_scenario.tests}
    assert dependencies["nccl_test_all_gather"]["start_post_comp"].test.name == "nccl_test_all_reduce"
    assert dependencies["nccl_test_reduce_scatter"]["start_post_comp"].test.name == "nccl_test_all_gather"
    assert dependencies["nccl_test_alltoall"]["start_post_comp"].test.name == "nccl_test_reduce_scatter"
    assert dependencies["nccl_test_bisection"]["start_post_comp"].test.name == "nccl_test_alltoall"
