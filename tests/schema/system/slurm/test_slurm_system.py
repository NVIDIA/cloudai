import pytest
from cloudai.schema.system import SlurmSystem
from cloudai.schema.system.slurm import SlurmNode, SlurmNodeState


@pytest.fixture
def slurm_system():
    nodes = [
        SlurmNode(name="nodeA001", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="nodeB001", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
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


def test_update_node_states_with_mocked_outputs(slurm_system):
    squeue_output = "nodeA001|root"
    sinfo_output = "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\nmain up infinite 1 idle nodeA001"
    slurm_system.update_node_states(squeue_output=squeue_output, sinfo_output=sinfo_output)
    assert slurm_system.partitions["main"][0].state == SlurmNodeState.IDLE
    assert slurm_system.partitions["main"][0].user == "root"
