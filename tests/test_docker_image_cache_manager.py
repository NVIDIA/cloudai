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

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from cloudai.util.docker_image_cache_manager import (
    DockerImageCacheManager,
    DockerImageCacheResult,
    PrerequisiteCheckResult,
)


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
@patch("os.access")
def test_ensure_docker_image_file_exists(mock_access, mock_exists, mock_is_file):
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")
    mock_is_file.return_value = True
    mock_exists.return_value = True
    result = manager.ensure_docker_image("/tmp/existing_file.sqsh", "docker_image.sqsh")
    assert result.success
    assert result.docker_image_path == Path("/tmp/existing_file.sqsh")
    assert result.message == "Docker image file path is valid: /tmp/existing_file.sqsh."


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
@patch("os.access")
def test_ensure_docker_image_url_cache_enabled(mock_access, mock_exists, mock_is_file):
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")
    mock_is_file.return_value = False
    mock_exists.return_value = True
    mock_access.return_value = True
    with patch.object(
        manager,
        "cache_docker_image",
        return_value=DockerImageCacheResult(
            True, Path("/fake/install/path/subdir/docker_image.sqsh"), "Docker image cached successfully."
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
@patch("cloudai.util.docker_image_cache_manager.DockerImageCacheManager._check_prerequisites")
def test_cache_docker_image(mock_check_prerequisites, mock_run, mock_access, mock_exists, mock_is_file):
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")

    # Test when cached file already exists
    mock_is_file.return_value = True
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    assert result.success
    assert result.docker_image_path == Path("/fake/install/path/image.tar.gz")
    assert result.message == "Cached Docker image already exists at /fake/install/path/image.tar.gz."

    # Test creating subdirectory when it doesn't exist
    mock_is_file.return_value = False
    mock_exists.side_effect = [True, False, True]  # install_path exists, subdir_path does not, install_path again
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")

    # Ensure prerequisites are always met for the following tests
    mock_check_prerequisites.return_value = PrerequisiteCheckResult(True, "All prerequisites are met.")

    # Reset the mock calls
    mock_run.reset_mock()
    mock_exists.side_effect = None

    # Test caching success with subprocess command (removal of default partition keyword)
    mock_is_file.return_value = False
    mock_exists.side_effect = [True, True, True, True, True]  # Ensure all path checks return True
    mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stderr="")
    result = manager.cache_docker_image("docker.io/hello-world", "image.tar.gz")
    mock_run.assert_called_once_with(
        "srun --export=ALL --partition=default enroot import -o /fake/install/path/image.tar.gz docker://docker.io/hello-world",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.success
    assert result.message == "Docker image cached successfully at /fake/install/path/image.tar.gz."

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
def test_uninstall_cached_image(mock_is_file, mock_unlink):
    # Mock setup
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")

    # Test successful removal
    mock_is_file.return_value = True
    result = manager.uninstall_cached_image("image.tar.gz")
    assert result.success
    assert result.message == "Cached Docker image removed successfully from /fake/install/path/image.tar.gz."
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
    assert result.message == "No cached Docker image found to remove at /fake/install/path/image.tar.gz."


@patch("shutil.which")
@patch("cloudai.util.docker_image_cache_manager.DockerImageCacheManager._check_docker_image_accessibility")
def test_check_prerequisites(mock_check_docker_image_accessibility, mock_which):
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")

    # Ensure enroot and srun are installed
    mock_which.side_effect = lambda x: x in ["enroot", "srun"]

    # Test all prerequisites met
    mock_check_docker_image_accessibility.return_value = PrerequisiteCheckResult(
        True, "Docker image URL is accessible."
    )
    result = manager._check_prerequisites("docker.io/hello-world")
    assert result.success
    assert result.message == "All prerequisites are met."

    # Test enroot not installed
    mock_which.side_effect = lambda x: x != "enroot"
    result = manager._check_prerequisites("docker.io/hello-world")
    assert not result.success
    assert result.message == "enroot are required for caching Docker images but are not installed."

    # Test srun not installed
    mock_which.side_effect = lambda x: x != "srun"
    result = manager._check_prerequisites("docker.io/hello-world")
    assert not result.success
    assert result.message == "srun are required for caching Docker images but are not installed."

    # Ensure enroot and srun are installed again
    mock_which.side_effect = lambda x: x in ["enroot", "srun"]

    # Test Docker image URL not accessible
    mock_check_docker_image_accessibility.return_value = PrerequisiteCheckResult(
        False, "Docker image URL not accessible."
    )
    result = manager._check_prerequisites("docker.io/hello-world")
    assert not result.success
    assert "Docker image URL not accessible." in result.message


@patch("subprocess.Popen")
@patch("shutil.which")
@patch("tempfile.TemporaryDirectory")
def test_check_docker_image_accessibility_with_enroot(mock_tempdir, mock_which, mock_popen):
    manager = DockerImageCacheManager(Path("/fake/install/path"), True, "default")

    # Ensure docker binary is not available
    mock_which.return_value = None

    # Mocking the TemporaryDirectory context manager
    mock_temp_dir = MagicMock()
    mock_tempdir.return_value.__enter__.return_value = mock_temp_dir
    temp_dir_path = Path("/fake/temp/dir")
    mock_temp_dir.__enter__.return_value = temp_dir_path

    # Mock Popen for enroot command with success scenario
    process_mock = MagicMock()
    process_mock.stderr.readline.side_effect = [
        b"Found all layers in cache\n",  # Simulate successful output
        b"",
        b"",
    ]
    process_mock.poll.side_effect = [None, None, 0]  # Simulate process running then finishing

    mock_popen.return_value = process_mock

    result = manager._check_docker_image_accessibility("docker.io/hello-world")
    assert result.success
    assert result.message == "Docker image URL, docker.io/hello-world, is accessible."

    # Verify the command contains the required keywords
    command = mock_popen.call_args[0][0]
    assert "enroot import -o" in command
    assert "docker://docker.io/hello-world" in command

    # Mock Popen for enroot command with failure scenario
    process_mock.reset_mock()
    process_mock.stderr.readline.side_effect = [
        b"[ERROR] URL https://nvcr.io/proxy_auth returned error code: 401 Unauthorized\n",  # Simulate 401 error output
        b"",
        b"",
    ]
    process_mock.poll.side_effect = [None, None, 0]  # Simulate process running then finishing

    mock_popen.return_value = process_mock

    result = manager._check_docker_image_accessibility("docker.io/hello-world")
    assert not result.success
    assert (
        "Failed to access Docker image URL: docker.io/hello-world. "
        "Error: [ERROR] URL https://nvcr.io/proxy_auth returned error code: 401 Unauthorized\n"
        "This error indicates that access to the Docker image URL is unauthorized. "
        "Please ensure you have the necessary permissions and have followed the instructions in the README "
        "for setting up your credentials correctly." in result.message.replace("\n\n", "\n")
    )

    # Verify the command contains the required keywords
    command = mock_popen.call_args[0][0]
    assert "enroot import -o" in command
    assert "docker://docker.io/hello-world" in command


@patch("pathlib.Path.is_file")
@patch("pathlib.Path.exists")
def test_ensure_docker_image_no_local_cache(mock_exists, mock_is_file):
    manager = DockerImageCacheManager(Path("/fake/install/path"), False, "default")

    mock_is_file.return_value = False
    mock_exists.return_value = True

    result = manager.ensure_docker_image("docker.io/hello-world", "docker_image.sqsh")

    assert result.success
    assert result.docker_image_path is not None
    assert str(result.docker_image_path) == "docker.io/hello-world"
    assert not result.docker_image_path.is_absolute()
    assert result.message == ""
