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

from concurrent.futures import Future
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai import InstallStatusResult, Parser, TestTemplate
from cloudai.__main__ import handle_dry_run_and_run, setup_logging
from cloudai._core.base_installer import BaseInstaller
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState

SLURM_TEST_SCENARIOS = [
    {"path": Path("conf/common/test_scenario/sleep.toml"), "expected_dirs_number": 4, "log_file": "sleep_debug.log"},
    {
        "path": Path("conf/common/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 5,
        "log_file": "ucc_test_debug.log",
    },
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")
    log_file = scenario.get("log_file", ".")
    log_file_path = tmp_path / log_file

    parser = Parser(Path("conf/common/system/example_slurm_cluster.toml"), Path("conf/common/test_template"))
    system, tests, test_scenario = parser.parse(Path("conf/common/test"), test_scenario_path)
    system.output_path = tmp_path
    assert test_scenario is not None, "Test scenario is None"
    setup_logging(log_file_path, "DEBUG")
    handle_dry_run_and_run("dry-run", system, tests, test_scenario)

    # Find the directory that was created for the test results
    results_output_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]

    # Assuming there's only one result directory created
    assert len(results_output_dirs) == 1, "No result directory found or multiple directories found."
    results_output = results_output_dirs[0]

    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"

    assert log_file_path.exists(), f"Log file {log_file_path} was not created"


@pytest.fixture
def slurm_system() -> SlurmSystem:
    nodes = [SlurmNode(name=f"node-0{i}", partition="main", state=SlurmNodeState.UNKNOWN_STATE) for i in range(33, 65)]
    backup_nodes = [
        SlurmNode(name=f"node0{i}", partition="backup", state=SlurmNodeState.UNKNOWN_STATE) for i in range(1, 9)
    ]

    system = SlurmSystem(
        name="test_system",
        install_path=Path("/fake/path"),
        output_path=Path("/fake/output"),
        default_partition="main",
        partitions={"main": nodes, "backup": backup_nodes},
    )
    return system


@pytest.fixture
def test_template_success() -> TestTemplate:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_success"
    template.install.return_value = InstallStatusResult(success=True)
    template.uninstall.return_value = InstallStatusResult(success=True)
    return template


@pytest.fixture
def test_template_failure() -> TestTemplate:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_failure"
    template.install.return_value = InstallStatusResult(success=False, message="Installation failed")
    template.uninstall.return_value = InstallStatusResult(success=False, message="Uninstallation failed")
    return template


def create_real_future(result):
    future = Future()
    future.set_result(result)
    return future


def extract_unique_test_templates(test_templates):
    unique_test_templates = {}
    for test_template in test_templates:
        template_name = test_template.name
        if template_name not in unique_test_templates:
            unique_test_templates[template_name] = test_template
    return list(unique_test_templates.values())


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_success(mock_executor: Mock, slurm_system: SlurmSystem, test_template_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_success])

    assert result.success
    assert result.message == "All test templates installed successfully."

    # Check if the template is installed
    assert installer.is_installed([test_template_success])


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_template_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to install."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_success(mock_executor: Mock, slurm_system: SlurmSystem, test_template_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_success])

    assert result.success
    assert result.message == "All test templates uninstalled successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_template_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to uninstall."

    # Check if the template is still installed
    assert installer.is_installed([test_template_failure])
