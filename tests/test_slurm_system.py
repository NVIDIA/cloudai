from typing import List
from unittest.mock import patch

import pytest
from cloudai.schema.system import SlurmSystem
from cloudai.schema.system.slurm import SlurmNode, SlurmNodeState


@pytest.fixture
def slurm_system():
    nodes = [
        SlurmNode(name="node-115", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-116", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-117", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-118", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-119", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-120", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-121", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node-122", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
    ]
    backup_nodes = [
        SlurmNode(name="node01", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node02", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node03", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node04", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node05", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node06", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node07", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="node08", partition="backup", state=SlurmNodeState.UNKNOWN_STATE),
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
    main    up    3:00:00      1  inval node-081
    main    up    3:00:00      5  drain node-[065-066,114,124-125]
    main    up    3:00:00      2   resv node-[034-035]
    main    up    3:00:00     88  alloc node-[033,036-064,067-080,082-113,115-123,126-128]
    backup    up   12:00:00     16  idle node[01-16]
    """
    node_user_map = {
        "": "user1",
        "node-033": "user2",
        "node-034": "user3",
        "node-056": "user3",
        "node-057": "user3",
        "node-058": "user3",
        "node-049": "user4",
        "node-050": "user4",
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
    for i in range(len(slurm_system.partitions["main"])):
        assert slurm_system.partitions["main"][i].state == SlurmNodeState.ALLOCATED
    for i in range(len(slurm_system.partitions["backup"])):
        assert slurm_system.partitions["backup"][i].state == SlurmNodeState.IDLE


@patch("cloudai.schema.system.SlurmSystem.get_squeue")
@patch("cloudai.schema.system.SlurmSystem.get_sinfo")
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
