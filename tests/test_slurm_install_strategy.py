from pathlib import Path

import pytest
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState
from cloudai.systems.slurm.strategy import SlurmInstallStrategy


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
def slurm_install_strategy(slurm_system: SlurmSystem) -> SlurmInstallStrategy:
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"docker_image_url": {"default": "http://example.com/docker_image"}}
    strategy = SlurmInstallStrategy(slurm_system, env_vars, cmd_args)
    return strategy


def test_install_path_attribute(slurm_install_strategy: SlurmInstallStrategy, slurm_system: SlurmSystem):
    assert slurm_install_strategy.install_path == slurm_system.install_path
