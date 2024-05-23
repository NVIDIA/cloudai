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
    system = SlurmSystem(
        name="test_system",
        install_path="/fake/path",
        output_path="/fake/output",
        default_partition="main",
        partitions={"main": nodes},
    )
    return system


def test_parse_squeue_output(slurm_system):
    squeue_output = "nodeA001|root\nnodeA002|user"
    expected_map = {"nodeA001": "root", "nodeA002": "user"}
    result = slurm_system.parse_squeue_output(squeue_output)
    assert result == expected_map


def test_parse_squeue_output_with_node_ranges_and_root_user(slurm_system):
    squeue_output = "nodeA[001-008],nodeB[001-008]|root"
    user_map = slurm_system.parse_squeue_output(squeue_output)

    expected_nodes = [f"nodeA{str(i).zfill(3)}" for i in range(1, 9)] + [f"nodeB{str(i).zfill(3)}" for i in range(1, 9)]
    expected_map = {node: "root" for node in expected_nodes}

    assert user_map == expected_map, "All nodes should be mapped to 'root'"


def test_parse_sinfo_output(slurm_system):
    sinfo_output = (
        "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n"
        "main up infinite 1 idle nodeA001\n"
        "main up infinite 1 idle nodeB001"
    )
    node_user_map = {"nodeA001": "root", "nodeB001": "user"}
    slurm_system.parse_sinfo_output(sinfo_output, node_user_map)
    assert slurm_system.partitions["main"][0].state == SlurmNodeState.IDLE
    assert slurm_system.partitions["main"][1].state == SlurmNodeState.IDLE
    
def test_parse_sinfo_output2(slurm_system):
    sinfo_output = """
    PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    main    up    3:00:00      1  inval node-081
    main    up    3:00:00      5  drain node-[065-066,114,124-125]
    main    up    3:00:00      2   resv node-[034-035]
    main    up    3:00:00     88  alloc node-[033,036-064,067-080,082-113,115-123,126-128]
    backup    up   12:00:00     16  alloc node-[01-16]
    """
    node_user_map = {'': 'user1', 'node-033': 'user2', 'node-[036-064': 'user3', '067-080': 'user3', '082-113': 'user3', '115-118]': 'user3', 'node-[119-123': 'user4', '126-128]': 'user4', 'node-01': 'user5', 'node-02': 'user5', 'node-03': 'user5', 'node-04': 'user5', 'node-05': 'user5', 'node-06': 'user5', 'node-07': 'user5', 'node-08': 'user5', 'node-09': 'user5', 'node-10': 'user5', 'node-11': 'user5', 'node-12': 'user5', 'node-13': 'user5', 'node-14': 'user5', 'node-15': 'user5', 'node-16': 'user5'}
    slurm_system.parse_sinfo_output(sinfo_output, node_user_map)
    assert slurm_system.partitions["main"][0].state == SlurmNodeState.ALLOCATED
    assert slurm_system.partitions["main"][1].state == SlurmNodeState.ALLOCATED


@patch("cloudai.schema.system.SlurmSystem.get_squeue")
@patch("cloudai.schema.system.SlurmSystem.get_sinfo")
def test_update_node_states_with_mocked_outputs(mock_get_sinfo, mock_get_squeue, slurm_system):
    mock_get_squeue.return_value = "nodeA001|root"
    mock_get_sinfo.return_value = "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n" "main up infinite 1 idle nodeA001"

    slurm_system.update_node_states()

    assert slurm_system.partitions["main"][0].state == SlurmNodeState.IDLE
    assert slurm_system.partitions["main"][0].user == "root"
