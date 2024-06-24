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
from unittest.mock import MagicMock, patch

import pytest
from cloudai import InstallStatusResult
from cloudai.schema.test_template.nccl_test.slurm_install_strategy import NcclTestSlurmInstallStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_install_strategy import (
    DatasetCheckResult,
    NeMoLauncherSlurmInstallStrategy,
)
from cloudai.schema.test_template.ucc_test.slurm_install_strategy import UCCTestSlurmInstallStrategy
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


@pytest.fixture
def mock_docker_image_cache_manager(slurm_system: SlurmSystem):
    mock = MagicMock()
    mock.cache_docker_images_locally = True
    mock.install_path = slurm_system.install_path
    mock.check_docker_image_exists.return_value = InstallStatusResult(success=False, message="Docker image not found")
    mock.ensure_docker_image.return_value = InstallStatusResult(success=True)
    mock.uninstall_cached_image.return_value = InstallStatusResult(success=True)
    return mock


class TestNcclTestSlurmInstallStrategy:
    @pytest.fixture
    def strategy(self, slurm_system, mock_docker_image_cache_manager) -> NcclTestSlurmInstallStrategy:
        strategy = NcclTestSlurmInstallStrategy(slurm_system, {}, {})
        strategy.docker_image_cache_manager = mock_docker_image_cache_manager
        return strategy

    def test_is_installed_locally(self, strategy: NcclTestSlurmInstallStrategy):
        expected_docker_image_path = str(Path(strategy.slurm_system.install_path) / "nccl-test" / "nccl_test.sqsh")

        result = strategy.is_installed()

        assert not result.success
        assert result.message == (
            "Docker image for NCCL test is not installed.\n"
            f"    - Expected path: {expected_docker_image_path}.\n"
            f"    - Error: Docker image not found"
        )

    def test_is_installed_remote(self, strategy: NcclTestSlurmInstallStrategy):
        strategy.docker_image_cache_manager.cache_docker_images_locally = False

        result = strategy.is_installed()

        assert not result.success
        assert result.message == (
            "Docker image for NCCL test is not accessible.\n" "    - Error: Docker image not found"
        )

    def test_install_success(self, strategy: NcclTestSlurmInstallStrategy):
        with patch.object(
            strategy.docker_image_cache_manager,
            "check_docker_image_exists",
            return_value=InstallStatusResult(success=True),
        ):
            result = strategy.install()
            assert result.success

    def test_uninstall_success(self, strategy: NcclTestSlurmInstallStrategy):
        result = strategy.uninstall()

        assert result.success


class TestNeMoLauncherSlurmInstallStrategy:
    @pytest.fixture
    def strategy(self, slurm_system, mock_docker_image_cache_manager) -> NeMoLauncherSlurmInstallStrategy:
        strategy = NeMoLauncherSlurmInstallStrategy(
            slurm_system,
            {},
            {
                "repository_url": {"default": "https://github.com/NVIDIA/NeMo-Framework-Launcher.git"},
                "repository_commit_hash": {"default": "cf411a9ede3b466677df8ee672bcc6c396e71e1a"},
                "docker_image_url": {"default": "nvcr.io/nvidian/nemofw-training:24.01.01"},
                "data_dir": {"default": "DATA_DIR"},
            },
        )
        strategy.docker_image_cache_manager = mock_docker_image_cache_manager
        return strategy

    def test_is_installed(self, strategy: NeMoLauncherSlurmInstallStrategy):
        with patch.object(
            strategy,
            "_check_datasets_on_nodes",
            return_value=DatasetCheckResult(success=True, nodes_without_datasets=[]),
        ):
            result = strategy.is_installed()
            assert not result.success
            assert (
                "The following components are missing:" in result.message
                and "Repository" in result.message
                and "Docker image" in result.message
            )

    def test_clone_repository_when_path_does_not_exist(self, strategy: NeMoLauncherSlurmInstallStrategy):
        subdir_path = Path(strategy.slurm_system.install_path) / strategy.SUBDIR_PATH
        repo_path = subdir_path / strategy.REPOSITORY_NAME
        assert not repo_path.exists()

        with patch("subprocess.run") as mock_run, patch("os.path.exists") as mock_exists:
            mock_run.return_value.returncode = 0
            mock_exists.side_effect = lambda path: path == str(subdir_path)
            strategy._clone_repository(str(subdir_path))
            strategy._install_requirements(str(subdir_path))

            mock_run.assert_any_call(
                ["git", "clone", strategy.repository_url, str(repo_path)], capture_output=True, text=True
            )
            mock_run.assert_any_call(
                ["git", "checkout", strategy.repository_commit_hash],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
            )

    def test_install_requirements(self, strategy: NeMoLauncherSlurmInstallStrategy):
        subdir_path = Path(strategy.slurm_system.install_path) / strategy.SUBDIR_PATH
        repo_path = subdir_path / strategy.REPOSITORY_NAME
        requirements_file = repo_path / "requirements.txt"
        repo_path.mkdir(parents=True, exist_ok=True)
        requirements_file.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            strategy._install_requirements(str(subdir_path))
            mock_run.assert_called_with(
                ["pip", "install", "-r", str(requirements_file)], capture_output=True, text=True
            )

    def test_clone_repository_when_path_exists(self, strategy: NeMoLauncherSlurmInstallStrategy):
        subdir_path = Path(strategy.slurm_system.install_path) / strategy.SUBDIR_PATH
        repo_path = subdir_path / strategy.REPOSITORY_NAME
        repo_path.mkdir(parents=True)

        with patch("subprocess.run") as mock_run, patch("os.path.exists") as mock_exists:
            mock_run.return_value.returncode = 0
            mock_exists.side_effect = lambda path: path in [str(subdir_path), str(repo_path)]
            strategy._clone_repository(str(subdir_path))

            # Ensure that the checkout command was run
            mock_run.assert_any_call(
                ["git", "checkout", strategy.repository_commit_hash],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
            )


class TestUCCTestSlurmInstallStrategy:
    @pytest.fixture
    def strategy(self, slurm_system, mock_docker_image_cache_manager) -> UCCTestSlurmInstallStrategy:
        strategy = UCCTestSlurmInstallStrategy(slurm_system, {}, {})
        strategy.docker_image_cache_manager = mock_docker_image_cache_manager
        return strategy

    def test_is_installed_locally(self, strategy: UCCTestSlurmInstallStrategy):
        expected_docker_image_path = str(Path(strategy.slurm_system.install_path) / "ucc-test" / "ucc_test.sqsh")

        result = strategy.is_installed()

        assert not result.success
        assert result.message == (
            "Docker image for UCC test is not installed.\n"
            f"    - Expected path: {expected_docker_image_path}.\n"
            f"    - Error: Docker image not found"
        )

    def test_is_installed_remote(self, strategy: UCCTestSlurmInstallStrategy):
        strategy.docker_image_cache_manager.cache_docker_images_locally = False

        result = strategy.is_installed()

        assert not result.success
        assert result.message == (
            "Docker image for UCC test is not accessible.\n" "    - Error: Docker image not found"
        )

    def test_install_success(self, strategy: UCCTestSlurmInstallStrategy):
        with patch.object(
            strategy.docker_image_cache_manager,
            "check_docker_image_exists",
            return_value=InstallStatusResult(success=True),
        ):
            result = strategy.install()
            assert result.success

    def test_uninstall_success(self, strategy: UCCTestSlurmInstallStrategy):
        result = strategy.uninstall()

        assert result.success
