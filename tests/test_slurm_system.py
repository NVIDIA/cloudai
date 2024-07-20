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

from typing import List
from unittest.mock import patch

import pytest
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState


@pytest.fixture
def slurm_system():
    nodes = [SlurmNode(name=f"node-0{i}", partition="main", state=SlurmNodeState.UNKNOWN_STATE) for i in range(33, 65)]
    backup_nodes = [
        SlurmNode(name=f"node0{i}", partition="backup", state=SlurmNodeState.UNKNOWN_STATE) for i in range(1, 9)
    ]

    system = SlurmSystem(
        name="test_system",
        install_path="/fake/path",
        output_path="/fake/output",
        default_partition="main",
        partitions={"main": nodes, "backup": backup_nodes},
    )
    return system


def test_parse_squeue_output(slurm_system):
    squeue_output = "nodeA001|root\nnodeA002|user"
    expected_map = {"nodeA001": "root", "nodeA002": "user"}
    result = slurm_system.parse_squeue_output(squeue_output)
    assert result == expected_map


def test_parse_squeue_output_with_node_ranges_and_root_user(slurm_system):
    squeue_output = "nodeA[001-008]|root\nnodeB[001-008]|root"
    user_map = slurm_system.parse_squeue_output(squeue_output)

    expected_nodes = [f"nodeA{str(i).zfill(3)}" for i in range(1, 9)] + [f"nodeB{str(i).zfill(3)}" for i in range(1, 9)]
    expected_map = {node: "root" for node in expected_nodes}

    assert user_map == expected_map, "All nodes should be mapped to 'root'"


def test_parse_sinfo_output(slurm_system):
    sinfo_output = """PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    main    up    3:00:00      1  inval node-036
    main    up    3:00:00      5  drain node-[045-046,059,061-062]
    main    up    3:00:00      2   resv node-[034-035]
    main    up    3:00:00     24  alloc node-[033,037-044,047-058,060,063-064]
    backup    up   12:00:00     8  idle node[01-08]
    empty_queue up infinite     0    n/a
    """
    node_user_map = {
        "": "user1",
        "node-033": "user2",
        "node-037": "user3",
        "node-038": "user3",
        "node-039": "user3",
        "node-040": "user3",
        "node-041": "user3",
        "node-042": "user4",
        "node-043": "user4",
        "node-044": "user4",
        "node01": "user5",
        "node02": "user5",
        "node03": "user5",
        "node04": "user5",
        "node05": "user5",
        "node06": "user5",
        "node07": "user5",
        "node08": "user5",
    }
    slurm_system.parse_sinfo_output(sinfo_output, node_user_map)
    inval_nodes = set(["node-036"])
    drain_nodes = set(["node-045", "node-046", "node-059", "node-061", "node-062"])
    resv_nodes = set(["node-034", "node-035"])
    for node in slurm_system.partitions["main"]:
        if node.name in inval_nodes:
            assert node.state == SlurmNodeState.INVALID_REGISTRATION
        elif node.name in drain_nodes:
            assert node.state == SlurmNodeState.DRAINED
        elif node.name in resv_nodes:
            assert node.state == SlurmNodeState.RESERVED
        else:
            print("node :", node)
            assert node.state == SlurmNodeState.ALLOCATED
    for node in slurm_system.partitions["backup"]:
        assert node.state == SlurmNodeState.IDLE


@patch("cloudai.systems.SlurmSystem.get_squeue")
@patch("cloudai.systems.SlurmSystem.get_sinfo")
def test_update_node_states_with_mocked_outputs(mock_get_sinfo, mock_get_squeue, slurm_system):
    mock_get_squeue.return_value = "node-115|user1"
    mock_get_sinfo.return_value = "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n" "main up infinite 1 idle node-115"

    slurm_system.update_node_states()
    for node in slurm_system.partitions["main"]:
        if node.name == "node-115":
            assert node.state == SlurmNodeState.IDLE
            assert node.user == "user1"

    mock_get_squeue.return_value = "node01|root"
    mock_get_sinfo.return_value = (
        "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n" "backup up infinite 1 allocated node01"
    )

    slurm_system.update_node_states()
    for node in slurm_system.partitions["backup"]:
        if node.name == "node01":
            assert node.state == SlurmNodeState.ALLOCATED
            assert node.user == "root"


@pytest.mark.parametrize(
    "node_list,expected_parsed_node_list",
    [
        ("node-[048-051]", ["node-048", "node-049", "node-050", "node-051"]),
        ("node-[055,114]", ["node-055", "node-114"]),
        ("node-[055,114],node-[056,115]", ["node-055", "node-114", "node-056", "node-115"]),
        ("", []),
        ("node-001", ["node-001"]),
        ("node[1-4]", ["node1", "node2", "node3", "node4"]),
        (
            "node-name[01-03,05-08,10]",
            [
                "node-name01",
                "node-name02",
                "node-name03",
                "node-name05",
                "node-name06",
                "node-name07",
                "node-name08",
                "node-name10",
            ],
        ),
    ],
)
def test_parse_node_list(node_list: str, expected_parsed_node_list: List[str], slurm_system):
    parsed_node_list = slurm_system.parse_node_list(node_list)
    assert parsed_node_list == expected_parsed_node_list
