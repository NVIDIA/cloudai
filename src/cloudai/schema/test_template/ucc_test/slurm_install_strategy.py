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

import os

from cloudai import InstallStatusResult
from cloudai.systems.slurm.strategy import SlurmInstallStrategy


class UCCTestSlurmInstallStrategy(SlurmInstallStrategy):
    """
    Install strategy for UCC tests on Slurm systems.

    Attributes
        SUBDIR_PATH (str): Subdirectory path where Docker images are stored.
        DOCKER_IMAGE_FILENAME (str): Name of the Docker image file.
    """

    SUBDIR_PATH = "ucc-test"
    DOCKER_IMAGE_FILENAME = "ucc_test.sqsh"

    def is_installed(self) -> InstallStatusResult:
        docker_image_result = self.docker_image_cache_manager.check_docker_image_exists(
            self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        )
        if docker_image_result.success:
            return InstallStatusResult(success=True)
        else:
            if self.docker_image_cache_manager.cache_docker_images_locally:
                expected_docker_image_path = os.path.join(
                    self.docker_image_cache_manager.install_path, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
                )
                return InstallStatusResult(
                    success=False,
                    message=(
                        f"Docker image for UCC test is not installed.\n"
                        f"    - Expected path: {expected_docker_image_path}.\n"
                        f"    - Error: {docker_image_result.message}"
                    ),
                )
            else:
                return InstallStatusResult(
                    success=False,
                    message=(
                        f"Docker image for UCC test is not accessible.\n" f"    - Error: {docker_image_result.message}"
                    ),
                )

    def install(self) -> InstallStatusResult:
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True)

        docker_image_result = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        )
        if not docker_image_result.success:
            return InstallStatusResult(
                success=False,
                message=(
                    "Failed to download and import the Docker image for UCC test. "
                    f"Error: {docker_image_result.message}"
                ),
            )

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        docker_image_result = self.docker_image_cache_manager.uninstall_cached_image(
            self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        )
        if not docker_image_result.success:
            return InstallStatusResult(
                success=False,
                message=("Failed to remove the Docker image for UCC test. Error: {docker_image_result.message}"),
            )

        return InstallStatusResult(success=True)
