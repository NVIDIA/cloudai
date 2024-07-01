#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock, patch

from cloudai.util.docker_image_cache_manager import (
    DockerImageCacheManager,
    DockerImageCacheResult,
    PrerequisiteCheckResult,
)


@patch("os.path.isfile")
@patch("os.path.exists")
@patch("os.access")
def test_ensure_docker_image_file_exists(mock_access, mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")
    mock_isfile.return_value = True
    mock_exists.return_value = True
    result = manager.ensure_docker_image("/tmp/existing_file.sqsh", "subdir", "docker_image.sqsh")
    assert result.success
    assert result.docker_image_path == "/tmp/existing_file.sqsh"
    assert result.message == "Docker image file path is valid: /tmp/existing_file.sqsh."


@patch("os.path.isfile")
@patch("os.path.exists")
@patch("os.access")
def test_ensure_docker_image_url_cache_enabled(mock_access, mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")
    mock_isfile.return_value = False
    mock_exists.return_value = True
    mock_access.return_value = True
    with patch.object(
        manager,
        "cache_docker_image",
        return_value=DockerImageCacheResult(
            True, "/fake/install/path/subdir/docker_image.sqsh", "Docker image cached successfully."
        ),
    ):
        result = manager.ensure_docker_image("docker.io/hello-world", "subdir", "docker_image.sqsh")
        assert result.success
        assert result.docker_image_path == "/fake/install/path/subdir/docker_image.sqsh"
        assert result.message == "Docker image cached successfully."


@patch("os.path.isfile")
@patch("os.path.exists")
@patch("os.access")
@patch("os.makedirs")
@patch("subprocess.run")
@patch("cloudai.util.docker_image_cache_manager.DockerImageCacheManager._check_prerequisites")
def test_cache_docker_image(mock_check_prerequisites, mock_run, mock_makedirs, mock_access, mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Test when cached file already exists
    mock_isfile.return_value = True
    result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
    assert result.success
    assert result.message == "Cached Docker image already exists at /fake/install/path/subdir/image.tar.gz."

    # Test creating subdirectory when it doesn't exist
    mock_isfile.return_value = False
    mock_exists.side_effect = [True, False, False]  # install_path exists, subdir_path does not
    with patch("os.makedirs") as mock_makedirs:
        result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
        mock_makedirs.assert_called_once_with("/fake/install/path/subdir")

    # Ensure prerequisites are always met for the following tests
    mock_check_prerequisites.return_value = PrerequisiteCheckResult(True, "All prerequisites are met.")

    # Reset the mock calls
    mock_run.reset_mock()
    mock_exists.side_effect = None

    # Test caching success with subprocess command (removal of default partition keyword)
    mock_isfile.return_value = False
    mock_exists.side_effect = [True, True, True, True, True]  # Ensure all path checks return True
    mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0)
    result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
    mock_run.assert_called_once_with(
        "srun --export=ALL --partition=default enroot import -o /fake/install/path/subdir/image.tar.gz docker://docker.io/hello-world",
        shell=True,
        check=True,
        capture_output=True,
    )
    assert result.success
    assert result.message == "Docker image cached successfully at /fake/install/path/subdir/image.tar.gz."

    # Test caching failure due to subprocess error
    mock_isfile.return_value = False
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
    result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
    assert not result.success


@patch("os.path.isfile")
@patch("os.path.exists")
@patch("os.access")
@patch("os.remove")
def test_remove_cached_image(mock_remove, mock_access, mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Test successful removal
    mock_isfile.return_value = True
    result = manager.remove_cached_image("subdir", "image.tar.gz")
    assert result.success
    assert result.message == "Cached Docker image removed successfully from /fake/install/path/subdir/image.tar.gz."
    mock_remove.assert_called_once_with("/fake/install/path/subdir/image.tar.gz")

    # Test failed removal due to OSError
    mock_remove.side_effect = OSError("Mocked OSError")
    result = manager.remove_cached_image("subdir", "image.tar.gz")
    assert not result.success
    assert "Failed to remove cached Docker image" in result.message

    # Test no file to remove
    mock_isfile.return_value = False
    result = manager.remove_cached_image("subdir", "image.tar.gz")
    assert result.success
    assert result.message == "No cached Docker image found to remove at /fake/install/path/subdir/image.tar.gz."


@patch("os.path.isfile")
@patch("os.path.exists")
@patch("os.access")
@patch("os.remove")
def test_uninstall_cached_image(mock_remove, mock_access, mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Test successful uninstallation and subdirectory removal
    mock_isfile.return_value = True
    mock_exists.return_value = True
    mock_access.return_value = True
    mock_listdir = patch("os.listdir", return_value=[]).start()
    result = manager.uninstall_cached_image("subdir", "image.tar.gz")
    assert result.success
    assert result.message in [
        "Cached Docker image uninstalled successfully from /fake/install/path/subdir.",
        "Subdirectory removed successfully: /fake/install/path/subdir.",
    ]
    mock_remove.assert_called_once_with("/fake/install/path/subdir/image.tar.gz")

    # Test successful uninstallation but subdirectory not empty
    mock_listdir.return_value = ["otherfile"]
    result = manager.uninstall_cached_image("subdir", "image.tar.gz")
    assert result.success
    assert result.message == "Cached Docker image uninstalled successfully from /fake/install/path/subdir."
    mock_remove.assert_called_with("/fake/install/path/subdir/image.tar.gz")

    # Test failed removal due to OSError
    mock_remove.side_effect = OSError("Mocked OSError")
    result = manager.uninstall_cached_image("subdir", "image.tar.gz")
    assert not result.success
    assert "Failed to remove cached Docker image" in result.message

    # Test no subdirectory to remove
    mock_isfile.return_value = False
    mock_exists.return_value = False
    result = manager.uninstall_cached_image("subdir", "image.tar.gz")
    assert result.success
    assert result.message == "Cached Docker image uninstalled successfully from /fake/install/path/subdir."

    # Cleanup
    patch.stopall()


@patch("shutil.which")
@patch("cloudai.util.docker_image_cache_manager.DockerImageCacheManager._check_docker_image_accessibility")
def test_check_prerequisites(mock_check_docker_image_accessibility, mock_which):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

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
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Ensure docker binary is not available
    mock_which.return_value = None

    # Mocking the TemporaryDirectory context manager
    mock_temp_dir = MagicMock()
    mock_tempdir.return_value.__enter__.return_value = mock_temp_dir
    temp_dir_path = "/fake/temp/dir"
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


@patch("os.path.isfile")
@patch("os.path.exists")
def test_check_docker_image_exists_with_only_cached_file(mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Simulate only the cached file being a valid file path
    mock_isfile.side_effect = lambda path: path == "/fake/install/path/subdir/docker_image.sqsh"
    mock_exists.side_effect = lambda path: path in [
        "/fake/install/path",
        "/fake/install/path/subdir",
        "/fake/install/path/subdir/docker_image.sqsh",
    ]

    result = manager.check_docker_image_exists("/tmp/non_existing_file.sqsh", "subdir", "docker_image.sqsh")
    assert result.success
    assert result.docker_image_path == "/fake/install/path/subdir/docker_image.sqsh"
    assert result.message == "Cached Docker image already exists at /fake/install/path/subdir/docker_image.sqsh."


@patch("os.path.isfile")
@patch("os.path.exists")
def test_check_docker_image_exists_with_both_valid_files(mock_exists, mock_isfile):
    manager = DockerImageCacheManager("/fake/install/path", True, "default")

    # Simulate both cached file and docker image URL being valid file paths
    mock_isfile.side_effect = lambda path: path in [
        "/tmp/existing_file.sqsh",
        "/fake/install/path/subdir/docker_image.sqsh",
    ]
    mock_exists.side_effect = lambda path: path in [
        "/tmp/existing_file.sqsh",
        "/fake/install/path/subdir/docker_image.sqsh",
    ]

    result = manager.check_docker_image_exists("/tmp/existing_file.sqsh", "subdir", "docker_image.sqsh")
    assert result.success
    assert result.docker_image_path == "/tmp/existing_file.sqsh"
    assert result.message == "Docker image file path is valid: /tmp/existing_file.sqsh."
