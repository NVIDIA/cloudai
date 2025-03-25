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

import re
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest

from cloudai import BaseJob
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState
from cloudai.systems.slurm.slurm_system import parse_node_list


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


def test_parse_sinfo_output(slurm_system: SlurmSystem) -> None:
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

    parts_by_name = {part.name: part for part in slurm_system.partitions}

    for node in parts_by_name["main"].slurm_nodes:
        if node.name in inval_nodes:
            assert node.state == SlurmNodeState.INVALID_REGISTRATION
        elif node.name in drain_nodes:
            assert node.state == SlurmNodeState.DRAINED
        elif node.name in resv_nodes:
            assert node.state == SlurmNodeState.RESERVED
        else:
            assert node.state == SlurmNodeState.ALLOCATED
    for node in parts_by_name["backup"].slurm_nodes:
        assert node.state == SlurmNodeState.IDLE


@patch("cloudai.systems.SlurmSystem.fetch_command_output")
def test_update_with_mocked_outputs(mock_fetch_command_output: Mock, slurm_system: SlurmSystem):
    mock_fetch_command_output.side_effect = [
        ("node-033|user1", ""),
        ("PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n" "main up infinite 1 idle node-033", ""),
    ]

    parts_by_name = {part.name: part for part in slurm_system.partitions}

    slurm_system.update()
    assert "node-033" in {node.name for node in parts_by_name["main"].slurm_nodes}
    for node in parts_by_name["main"].slurm_nodes:
        if node.name == "node-033":
            assert node.state == SlurmNodeState.IDLE
            assert node.user == "user1"

    mock_fetch_command_output.side_effect = [
        ("node01|root", ""),
        ("PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n" "backup up infinite 1 allocated node01", ""),
    ]

    slurm_system.update()
    for node in parts_by_name["backup"].slurm_nodes:
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
def test_parse_node_list(node_list: str, expected_parsed_node_list: List[str]):
    parsed_node_list = parse_node_list(node_list)
    assert parsed_node_list == expected_parsed_node_list


@pytest.fixture
def grouped_nodes() -> dict[SlurmNodeState, list[SlurmNode]]:
    """
    Helper function to set up a mock Slurm system with nodes and their states.
    """
    partition_name = "main"

    grouped_nodes = {
        SlurmNodeState.IDLE: [
            SlurmNode(name="node01", partition=partition_name, state=SlurmNodeState.IDLE),
            SlurmNode(name="node02", partition=partition_name, state=SlurmNodeState.IDLE),
        ],
        SlurmNodeState.COMPLETING: [
            SlurmNode(name="node04", partition=partition_name, state=SlurmNodeState.COMPLETING)
        ],
        SlurmNodeState.ALLOCATED: [SlurmNode(name="node05", partition=partition_name, state=SlurmNodeState.ALLOCATED)],
    }

    return grouped_nodes


def test_get_available_nodes_exceeding_limit_no_callstack(
    slurm_system: SlurmSystem, grouped_nodes: Dict[SlurmNodeState, List[SlurmNode]], caplog
):
    group_name = "group1"
    partition_name = "main"
    num_nodes = 5

    slurm_system.get_available_nodes_from_group(partition_name, group_name, num_nodes)

    log_message = "CloudAI is requesting 5 nodes from the group 'group1', but only 0 nodes are available."
    assert log_message in caplog.text


def test_allocate_nodes_max_avail(slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]):
    group_name = "group_name"

    available_nodes = slurm_system.allocate_nodes(grouped_nodes, "max_avail", group_name)
    expected_node_names = [
        grouped_nodes[SlurmNodeState.IDLE][0].name,
        grouped_nodes[SlurmNodeState.IDLE][1].name,
        grouped_nodes[SlurmNodeState.COMPLETING][0].name,
    ]
    returned_node_names = [node.name for node in available_nodes]

    assert set(returned_node_names) == set(
        expected_node_names
    ), "Should return all available nodes except ALLOCATED nodes"
    allocated_node_name = grouped_nodes[SlurmNodeState.ALLOCATED][0].name
    assert allocated_node_name not in returned_node_names, "ALLOCATED node should not be included"


def test_allocate_nodes_num_nodes_integers(
    slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]
):
    group_name = "group_name"

    available_nodes = slurm_system.allocate_nodes(grouped_nodes, 1, group_name)
    expected_node_names = [grouped_nodes[SlurmNodeState.IDLE][0].name]

    returned_node_names = [node.name for node in available_nodes]

    assert len(returned_node_names) == len(expected_node_names), "Should return 1 available node"


def test_allocate_nodes_exceeding_limit(
    slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]
):
    group_name = "group_name"
    num_nodes = 5
    available_nodes = 4

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"CloudAI is requesting {num_nodes} nodes from the group '{group_name}', but only "
            f"{available_nodes} nodes are available. Please review the available nodes in the system "
            f"and ensure there are enough resources to meet the requested node count. Additionally, "
            f"verify that the system can accommodate the number of nodes required by the test scenario."
        ),
    ):
        slurm_system.allocate_nodes(grouped_nodes, num_nodes, group_name)


@pytest.mark.parametrize(
    "stdout,stderr,is_completed",
    [
        ("COMPLETED", "", True),
        ("FAILED", "", True),
        ("CANCELLED", "", True),
        ("TIMEOUT", "", True),
        ("RUNNING", "", False),
        ("PENDING", "", False),
        ("COMPLETED RUNNING", "", False),
        ("RUNNING COMPLETED", "", False),
        ("COMPLETED COMPLETED", "", True),
        ("", "error", False),
    ],
)
def test_is_job_completed(stdout: str, stderr: str, is_completed: bool, slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, stderr))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    if stderr:
        with pytest.raises(RuntimeError):
            slurm_system.is_job_completed(job)
    else:
        assert slurm_system.is_job_completed(job) is is_completed


@pytest.mark.parametrize(
    "stdout,stderr,is_running",
    [
        ("RUNNING", "", True),
        ("PENDING", "", False),
        ("COMPLETED", "", False),
        ("FAILED", "", False),
        ("CANCELLED", "", False),
        ("TIMEOUT", "", False),
        ("", "error", False),
    ],
)
def test_is_job_running(stdout: str, stderr: str, is_running: bool, slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, stderr))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    if stderr:
        with pytest.raises(RuntimeError):
            slurm_system.is_job_running(job)
    else:
        assert slurm_system.is_job_running(job) is is_running


def test_is_job_running_with_retries(slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    command = f"sacct -j {job.id} --format=State --noheader"

    pp = Mock()
    pp.communicate = Mock(side_effect=[("", "Socket timed out"), ("", "slurm_load_jobs error"), ("RUNNING", "")])
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    assert slurm_system.is_job_running(job, retry_threshold=3) is True
    assert slurm_system.cmd_shell.execute.call_count == 3
    slurm_system.cmd_shell.execute.assert_called_with(command)


def test_is_job_running_exceeds_retries(slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    command = f"sacct -j {job.id} --format=State --noheader"

    # test known error in srderr
    pp = Mock()
    pp.communicate = Mock(return_value=("", "Socket timed out"))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)
    with pytest.raises(RuntimeError):
        slurm_system.is_job_running(job)
    assert slurm_system.cmd_shell.execute.call_count == 3
    slurm_system.cmd_shell.execute.assert_called_with(command)

    # test unknown error in stderr
    pp.communicate = Mock(return_value=("", "FAILED"))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)
    with pytest.raises(RuntimeError):
        slurm_system.is_job_running(job, retry_threshold=3)
    assert slurm_system.cmd_shell.execute.call_count == 1


def test_model_dump(slurm_system: SlurmSystem):
    sys_dict = slurm_system.model_dump()
    assert type(sys_dict["install_path"]) is str
    assert sys_dict["install_path"] == str(slurm_system.install_path)
    assert type(sys_dict["output_path"]) is str
    assert sys_dict["output_path"] == str(slurm_system.output_path)
    assert "cmd_shell" not in sys_dict
    recreated = SlurmSystem.model_validate(sys_dict)
    assert recreated.model_dump() == sys_dict


def test_default_partition_is_required():
    with pytest.raises(ValueError):
        SlurmSystem(name="", install_path=Path.cwd(), output_path=Path.cwd(), partitions=[])  # type: ignore


class TestParseNodes:
    def test_single_node(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01"])
        assert nodes == ["node01"]

    def test_two_nodes(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01", "node02"])
        assert nodes == ["node01", "node02"]

    def test_range(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node0[1-3]"])
        assert nodes == ["node01", "node02", "node03"]

    def test_with_commas(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01,node02,node03"])
        assert nodes == ["node01", "node02", "node03"]

    @pytest.mark.parametrize("spec", ["part:", "part:group", "unknown:grp:1", "main:unknown:1"])
    def test_colon_invalid_syntax(self, slurm_system: SlurmSystem, spec: str):
        with pytest.raises(ValueError):
            slurm_system.parse_nodes([spec])
