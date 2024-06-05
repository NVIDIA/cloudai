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

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from cloudai._core.base_installer import BaseInstaller
from cloudai._core.install_status_result import InstallStatusResult
from cloudai._core.test_template import TestTemplate
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


@pytest.fixture
def test_template_success():
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_success"
    template.install.return_value = InstallStatusResult(success=True)
    template.uninstall.return_value = InstallStatusResult(success=True)
    return template


@pytest.fixture
def test_template_failure():
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_failure"
    template.install.return_value = InstallStatusResult(success=False, message="Installation failed")
    template.uninstall.return_value = InstallStatusResult(success=False, message="Uninstallation failed")
    return template


def create_real_future(result):
    future = Future()
    future.set_result(result)
    return future


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_success(mock_executor, slurm_system, test_template_success):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_success])

    assert result.success
    assert result.message == "All test templates installed successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_failure(mock_executor, slurm_system, test_template_failure):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to install."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_success(mock_executor, slurm_system, test_template_success):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_success])

    assert result.success
    assert result.message == "All test templates uninstalled successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_failure(mock_executor, slurm_system, test_template_failure):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to uninstall."
