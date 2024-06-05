from concurrent.futures import Future
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai._core.base_installer import BaseInstaller
from cloudai._core.install_status_result import InstallStatusResult
from cloudai._core.test import Test
from cloudai._core.test_template import TestTemplate
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState


@pytest.fixture
def slurm_system() -> SlurmSystem:
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


@pytest.fixture
def test_success() -> Test:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_success"
    template.install.return_value = InstallStatusResult(success=True)
    template.uninstall.return_value = InstallStatusResult(success=True)
    test = Mock(spec=Test)
    test.test_template = template
    return test


@pytest.fixture
def test_failure() -> Test:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_failure"
    template.install.return_value = InstallStatusResult(success=False, message="Installation failed")
    template.uninstall.return_value = InstallStatusResult(success=False, message="Uninstallation failed")
    test = Mock(spec=Test)
    test.test_template = template
    return test


def create_real_future(result):
    future = Future()
    future.set_result(result)
    return future


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_success(mock_executor: Mock, slurm_system: SlurmSystem, test_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_success.test_template.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_success])

    assert result.success
    assert result.message == "All test templates installed successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_failure.test_template.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_failure])

    assert not result.success
    assert result.message == "Some test templates failed to install."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_success(mock_executor: Mock, slurm_system: SlurmSystem, test_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_success.test_template.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_success])

    assert result.success
    assert result.message == "All test templates uninstalled successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_failure.test_template.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_failure])

    assert not result.success
    assert result.message == "Some test templates failed to uninstall."
