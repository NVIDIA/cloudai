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

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cloudai.systems.slurm.docker_image_cache_manager import (
    DockerImageCacheManager,
    DockerImageCacheResult,
    PrerequisiteCheckResult,
)
from cloudai.systems.slurm.slurm_system import SlurmSystem


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
@patch("os.access")
def test_ensure_docker_image_file_exists(mock_access, mock_exists, mock_is_file, slurm_system: SlurmSystem):
    manager = DockerImageCacheManager(slurm_system)
    mock_is_file.return_value = True
    mock_exists.return_value = True
    result = manager.ensure_docker_image("/tmp/existing_file.sqsh", "docker_image.sqsh")
    assert result.success
    assert result.docker_image_path == Path("/tmp/existing_file.sqsh")
    assert result.message == "Docker image file path is valid: /tmp/existing_file.sqsh."


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
@patch("os.access")
def test_ensure_docker_image_url_cache_enabled(mock_access, mock_exists, mock_is_file, slurm_system: SlurmSystem):
    manager = DockerImageCacheManager(slurm_system)
    mock_is_file.return_value = False
    mock_exists.return_value = True
    mock_access.return_value = True
    with patch.object(
        manager,
        "cache_docker_image",
        return_value=DockerImageCacheResult(
            True,
            Path("/fake/install/path/subdir/docker_image.sqsh"),
            "Docker image cached successfully.",
        ),
    ):
        result = manager.ensure_docker_image("docker.io/hello-world", "docker_image.sqsh")
        assert result.success
        assert result.docker_image_path == Path("/fake/install/path/subdir/docker_image.sqsh")
        assert result.message == "Docker image cached successfully."


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
@patch("os.access")
@patch("subprocess.run")
@patch("cloudai.systems.slurm.docker_image_cache_manager.DockerImageCacheManager._check_prerequisites")
def test_cache_docker_image(
    mock_check_prerequisites, mock_run, mock_access, mock_exists, mock_is_file, slurm_system: SlurmSystem
):
    manager = DockerImageCacheManager(slurm_system)

    # Test when cached file already exists
    mock_is_file.return_value = True
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    assert result.success
    assert result.docker_image_path == slurm_system.install_path / "image.tar.gz"
    assert result.message == f"Cached Docker image already exists at {slurm_system.install_path}/image.tar.gz."

    # Test creating subdirectory when it doesn't exist
    mock_is_file.return_value = False
    mock_exists.side_effect = [
        True,
        False,
        True,
    ]  # install_path exists, subdir_path does not, install_path again
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")

    # Ensure prerequisites are always met for the following tests
    mock_check_prerequisites.return_value = PrerequisiteCheckResult(True, "All prerequisites are met.")

    # Reset the mock calls
    mock_run.reset_mock()
    mock_exists.side_effect = None

    # Test caching success with subprocess command (removal of default partition keyword)
    mock_is_file.return_value = False
    mock_exists.side_effect = [
        True,
        True,
        True,
        True,
        True,
    ]  # Ensure all path checks return True
    mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stderr="")
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")

    assert mock_run.call_count == 1
    actual_command = mock_run.call_args[0][0]
    assert f"srun --export=ALL --partition={slurm_system.default_partition}" in actual_command
    assert "--ntasks=1" in actual_command
    assert "-N1" in actual_command
    assert "--job-name=CloudAI_install_docker_image_" in actual_command
    assert f"enroot import -o {slurm_system.install_path}/image.tar.gz docker://docker.io/hello-world" in actual_command
    assert mock_run.call_args[1] == {"shell": True, "check": True, "capture_output": True, "text": True}

    assert result.success
    assert result.message == f"Docker image cached successfully at {slurm_system.install_path}/image.tar.gz."

    # Test caching failure due to subprocess error
    mock_is_file.return_value = False
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    assert not result.success

    # Test caching failure due to disk-related errors
    mock_is_file.return_value = False
    mock_run.side_effect = None
    mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=1, stderr="Disk quota exceeded\n")
    mock_exists.side_effect = [True, True, True, True, True]
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    assert not result.success
    assert "Disk quota exceeded" in result.message

    mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=1, stderr="Write error\n")
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    assert not result.success
    assert "Write error" in result.message


@patch("pathlib.Path.unlink")
@patch("pathlib.Path.is_file")
def test_uninstall_cached_image(mock_is_file, mock_unlink, slurm_system: SlurmSystem):
    # Mock setup
    manager = DockerImageCacheManager(slurm_system)

    # Test successful removal
    mock_is_file.return_value = True
    result = manager.uninstall_cached_image("image.tar.gz")
    assert result.success
    assert result.message == f"Cached Docker image removed successfully from {slurm_system.install_path}/image.tar.gz."
    mock_unlink.assert_called_once()

    # Test failed removal due to OSError
    mock_unlink.side_effect = OSError("Mocked OSError")
    result = manager.uninstall_cached_image("image.tar.gz")
    assert not result.success
    assert "Failed to remove cached Docker image" in result.message

    # Test no file to remove
    mock_is_file.return_value = False
    result = manager.uninstall_cached_image("image.tar.gz")
    assert result.success
    assert result.message == f"No cached Docker image found to remove at {slurm_system.install_path}/image.tar.gz."


@patch("shutil.which")
def test_check_prerequisites(mock_which, slurm_system: SlurmSystem):
    manager = DockerImageCacheManager(slurm_system)

    # Ensure enroot and srun are installed
    mock_which.side_effect = lambda x: x in ["enroot", "srun"]

    # Test all prerequisites met
    result = manager._check_prerequisites()
    assert result.success
    assert result.message == "All prerequisites are met."

    # Test srun not installed
    mock_which.side_effect = lambda x: x != "srun"
    result = manager._check_prerequisites()
    assert not result.success
    assert result.message == "srun are required for caching Docker images but are not installed."


def test_ensure_docker_image_no_local_cache(slurm_system: SlurmSystem):
    slurm_system.cache_docker_images_locally = False
    manager = DockerImageCacheManager(slurm_system)

    result = manager.ensure_docker_image("docker.io/hello-world", "docker_image.sqsh")

    assert result.success
    assert result.docker_image_path is None
    assert result.message == ""


@pytest.mark.parametrize(
    "account, supports_gpu_directives", [(None, False), ("test_account", True), (None, False), ("test_account", True)]
)
def test_docker_import_with_extra_system_config(
    slurm_system: SlurmSystem, account: str | None, supports_gpu_directives: bool | None
):
    slurm_system.account = account
    slurm_system.supports_gpu_directives_cache = supports_gpu_directives
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)

    manager = DockerImageCacheManager(slurm_system)
    manager._check_prerequisites = lambda: PrerequisiteCheckResult(True, "All prerequisites are met.")

    with patch("subprocess.run") as mock_run:
        res = manager.cache_docker_image("docker.io/hello-world", "docker_image.sqsh")
        assert res.success

    assert mock_run.call_count == 1

    actual_command = mock_run.call_args[0][0]

    expected_prefix = f"srun --export=ALL --partition={slurm_system.default_partition}"
    assert expected_prefix in actual_command
    assert "-N1" in actual_command

    if account:
        assert f"--account={account}" in actual_command
        assert f"--job-name={account}-CloudAI_install_docker_image." in actual_command
    else:
        assert "--job-name=CloudAI_install_docker_image_" in actual_command

    if supports_gpu_directives:
        assert "--gres=gpu:1" in actual_command

    assert (
        f"enroot import -o {slurm_system.install_path}/docker_image.sqsh docker://docker.io/hello-world"
        in actual_command
    )

    assert mock_run.call_args[1] == {"shell": True, "check": True, "capture_output": True, "text": True}


def test_docker_import_with_extra_srun_args(slurm_system: SlurmSystem):
    slurm_system.extra_srun_args = "--reservation test_reservation --w hgx-isr1-pre-09"
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)

    manager = DockerImageCacheManager(slurm_system)
    manager._check_prerequisites = lambda: PrerequisiteCheckResult(True, "All prerequisites are met.")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stderr="")
        res = manager.cache_docker_image("docker.io/hello-world", "docker_image.sqsh")
        assert res.success

    assert mock_run.call_count == 1
    actual_command = mock_run.call_args[0][0]

    # Check that extra_srun_args are included
    assert "--reservation test_reservation" in actual_command
    assert "--w hgx-isr1-pre-09" in actual_command

    # Check that the command still has the basic structure
    expected_prefix = f"srun --export=ALL --partition={slurm_system.default_partition}"
    assert expected_prefix in actual_command
    assert "enroot import -o" in actual_command
