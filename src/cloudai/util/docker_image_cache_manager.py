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

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class PrerequisiteCheckResult:
    """
    Class representing the result of a prerequisite check.

    Attributes
        success (bool): Indicates whether the prerequisite check was successful.
        message (str): A message providing additional information about the result.
    """

    def __init__(self, success: bool, message: str = "") -> None:
        """
        Initialize the PrerequisiteCheckResult.

        Args:
            success (bool): Indicates whether the prerequisite check was successful.
            message (str): A message providing additional information about the result.
        """
        self.success = success
        self.message = message

    def __bool__(self):
        """
        Return the success status as a boolean.

        Returns
            bool: True if the check was successful, False otherwise.
        """
        return self.success

    def __str__(self):
        """
        Return the message as a string.

        Returns
            str: The message providing additional information about the result.
        """
        return self.message


class DockerImageCacheResult:
    """
    Class representing the result of a Docker image caching operation.

    Attributes
        success (bool): Indicates whether the operation was successful.
        docker_image_path (Path): The path to the Docker image.
        message (str): A message providing additional information about the result.
    """

    def __init__(self, success: bool, docker_image_path: Optional[Path] = None, message: str = "") -> None:
        """
        Initialize the DockerImageCacheResult.

        Args:
            success (bool): Indicates whether the operation was successful.
            docker_image_path (Path): The path to the Docker image.
            message (str): A message providing additional information about the result.
        """
        self.success = success
        self.docker_image_path = docker_image_path
        self.message = message

    def __bool__(self):
        """
        Return the success status as a boolean.

        Returns
            bool: True if the operation was successful, False otherwise.
        """
        return self.success

    def __str__(self):
        """
        Return the message as a string.

        Returns
            str: The message providing additional information about the result.
        """
        return self.message


class DockerImageCacheManager:
    """
    Manages the caching of Docker images for installation strategies.

    Attributes
        install_path (Path): The base installation path.
        cache_docker_images_locally (bool): Whether to cache Docker image files locally.
        partition_name (str): The partition name to use in the srun command.
    """

    def __init__(self, install_path: Path, cache_docker_images_locally: bool, partition_name: str) -> None:
        self.install_path = install_path
        self.cache_docker_images_locally = cache_docker_images_locally
        self.partition_name = partition_name

    def ensure_docker_image(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        """
        Ensure the Docker image exists by checking and optionally caching it.

        Args:
            docker_image_url (str): URL or file path of the Docker image.
            docker_image_filename (str): Docker image filename.

        Returns:
            DockerImageCacheResult: Result of ensuring the Docker image exists.
        """
        image_check_result = self.check_docker_image_exists(docker_image_url, docker_image_filename)
        if image_check_result.success:
            return image_check_result

        if self.cache_docker_images_locally:
            return self.cache_docker_image(docker_image_url, docker_image_filename)

        return image_check_result

    def check_docker_image_exists(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        """
        Check if the Docker image exists without caching it.

        Args:
            docker_image_url (str): URL or file path of the Docker image.
            docker_image_filename (str): Docker image filename.

        Returns:
            DockerImageCacheResult: Result of the Docker image existence check.
        """
        logging.debug(
            f"Checking if Docker image exists: docker_image_url={docker_image_url}, subdir_name={self.install_path}, "
            f"docker_image_filename={docker_image_filename}, "
            f"cache_docker_images_locally={self.cache_docker_images_locally}"
        )

        # If not caching locally, return True. Defer checking URL accessibility to srun.
        if not self.cache_docker_images_locally:
            return DockerImageCacheResult(True, Path(docker_image_url), "")

        docker_image_path = Path(docker_image_url)
        if docker_image_path.is_file() and docker_image_path.exists():
            return DockerImageCacheResult(
                True,
                docker_image_path.absolute(),
                f"Docker image file path is valid: {docker_image_url}.",
            )

        # Check if the cache file exists
        if not self.install_path.exists():
            message = f"Install path {self.install_path.absolute()} does not exist."
            logging.debug(message)
            return DockerImageCacheResult(False, Path(), message)

        docker_image_path = self.install_path / docker_image_filename
        if docker_image_path.is_file() and docker_image_path.exists():
            message = f"Cached Docker image already exists at {docker_image_path}."
            logging.debug(message)
            return DockerImageCacheResult(True, docker_image_path.absolute(), message)

        message = f"Docker image does not exist at the specified path: {docker_image_path}."
        logging.debug(message)
        return DockerImageCacheResult(False, Path(), message)

    def cache_docker_image(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        """
        Cache the Docker image locally using enroot import.

        Args:
            docker_image_url (str): URL of the Docker image.
            docker_image_filename (str): Docker image filename.

        Returns:
            DockerImageCacheResult: Result of the Docker image caching operation.
        """
        docker_image_path = self.install_path / docker_image_filename

        if docker_image_path.is_file():
            success_message = f"Cached Docker image already exists at {docker_image_path}."
            logging.info(success_message)
            return DockerImageCacheResult(True, docker_image_path.absolute(), success_message)

        if not self.install_path.exists():
            error_message = f"Install path {self.install_path.absolute()} does not exist."
            logging.error(error_message)
            return DockerImageCacheResult(False, Path(), error_message)

        prerequisite_check = self._check_prerequisites()
        if not prerequisite_check:
            logging.error(f"Prerequisite check failed: {prerequisite_check.message}")
            return DockerImageCacheResult(False, Path(), prerequisite_check.message)

        if not os.access(self.install_path, os.W_OK):
            error_message = f"No permission to write in install path {self.install_path}."
            logging.error(error_message)
            return DockerImageCacheResult(False, Path(), error_message)

        enroot_import_cmd = (
            f"srun --export=ALL --partition={self.partition_name} "
            f"enroot import -o {docker_image_path} docker://{docker_image_url}"
        )
        logging.debug(f"Importing Docker image: {enroot_import_cmd}")

        try:
            p = subprocess.run(enroot_import_cmd, shell=True, check=True, capture_output=True, text=True)

            if "Disk quota exceeded" in p.stderr or "Write error" in p.stderr:
                error_message = (
                    f"Failed to cache Docker image {docker_image_url}. Command: {enroot_import_cmd}. "
                    f"Error: '{p.stderr}'\n\n"
                    "This error indicates a disk-related issue. Please check if the disk is full or not usable. "
                    "If the disk is full, consider using a different disk or removing unnecessary files."
                )
                logging.error(error_message)
                return DockerImageCacheResult(False, Path(), error_message)

            success_message = f"Docker image cached successfully at {docker_image_path}."
            logging.debug(success_message)
            logging.debug(f"Command used: {enroot_import_cmd}, stdout: {p.stdout}, stderr: {p.stderr}")
            return DockerImageCacheResult(True, docker_image_path.absolute(), success_message)
        except subprocess.CalledProcessError as e:
            error_message = (
                f"Failed to import Docker image {docker_image_url}. Command: {enroot_import_cmd}. Error: {e.stderr}"
            )
            logging.error(error_message)
            return DockerImageCacheResult(False, message=error_message)

    def _check_prerequisites(self) -> PrerequisiteCheckResult:
        """
        Check prerequisites for caching Docker image.

        Returns:
            PrerequisiteCheckResult: Result of the prerequisite check.
        """
        required_binaries = ["enroot", "srun"]
        missing_binaries = [binary for binary in required_binaries if not shutil.which(binary)]

        if missing_binaries:
            missing_binaries_str = ", ".join(missing_binaries)
            logging.error(f"{missing_binaries_str} are required for caching Docker images but are not installed.")
            return PrerequisiteCheckResult(
                False,
                f"{missing_binaries_str} are required for caching Docker images but are not installed.",
            )

        return PrerequisiteCheckResult(True, "All prerequisites are met.")

    def uninstall_cached_image(self, docker_image_filename: str) -> DockerImageCacheResult:
        """
        Remove an existing cached Docker image.

        Args:
            docker_image_filename (str): Docker image filename.

        Returns:
            DockerImageCacheResult: Result of the removal operation.
        """
        docker_image_path = self.install_path / docker_image_filename
        if docker_image_path.is_file():
            try:
                docker_image_path.unlink()
                success_message = f"Cached Docker image removed successfully from {docker_image_path}."
                logging.info(success_message)
                return DockerImageCacheResult(True, docker_image_path.absolute(), success_message)
            except OSError as e:
                error_message = f"Failed to remove cached Docker image at {docker_image_path}. Error: {e}"
                logging.error(error_message)
                return DockerImageCacheResult(False, docker_image_path, error_message)
        success_message = f"No cached Docker image found to remove at {docker_image_path}."
        logging.warning(success_message)
        return DockerImageCacheResult(True, docker_image_path.absolute(), success_message)
