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

import os
import shutil
import subprocess

from cloudai._core.install_status_result import InstallStatusResult
from cloudai.systems.slurm.strategy import SlurmInstallStrategy
from cloudai.util import CommandShell


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
        docker_image_path = os.path.join(self.install_path, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME)
        if os.path.isfile(docker_image_path):
            return InstallStatusResult(success=True)
        else:
            return InstallStatusResult(
                success=False,
                message=(
                    "Docker image for UCC test is not installed. "
                    f"Tried to find Docker image at: {docker_image_path}. "
                    "Please ensure the Docker image is present at the specified location."
                ),
            )

    def install(self) -> InstallStatusResult:
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True)

        docker_image_dir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        os.makedirs(docker_image_dir_path, exist_ok=True)
        docker_image_path = os.path.join(docker_image_dir_path, self.DOCKER_IMAGE_FILENAME)

        # Remove existing Docker image if it exists
        shell = CommandShell()
        remove_cmd = f"rm -f {docker_image_path}"
        process = shell.execute(remove_cmd)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return InstallStatusResult(
                success=False,
                message=(
                    f"Failed to remove existing Docker image at {docker_image_path}. "
                    "UCC tests tried to download a new Docker image, but an existing Docker image was found. "
                    "CloudAI tried to remove it but failed. "
                    f"Command run: {remove_cmd}. "
                    f"Error: {stderr}"
                ),
            )

        # Import new Docker image using enroot
        docker_image_url_info = self.cmd_args.get("docker_image_url")
        docker_image_url = docker_image_url_info.get("default") if docker_image_url_info else None
        if docker_image_url is None:
            return InstallStatusResult(
                success=False,
                message=(
                    "docker_image_url not found in the test schema or its value is not valid. "
                    "You should have a valid Docker image URL to the UCC test Docker image."
                ),
            )

        enroot_import_cmd = (
            f"srun"
            f" --export=ALL"
            f" --partition={self.slurm_system.default_partition}"
            f" enroot import -o {docker_image_path} docker://{docker_image_url}"
        )

        try:
            subprocess.run(enroot_import_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            return InstallStatusResult(
                success=False,
                message=(
                    f"Failed to import Docker image from {docker_image_url}. "
                    "CloudAI failed to import the Docker image. "
                    f"Command run: {enroot_import_cmd}. "
                    f"Error: {e}. "
                    "Please check the Docker image URL and ensure that it is accessible and set up "
                    "with valid credentials."
                ),
            )

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        docker_image_path = os.path.join(self.install_path, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME)

        if os.path.isfile(docker_image_path):
            try:
                os.remove(docker_image_path)
            except OSError as e:
                return InstallStatusResult(
                    success=False,
                    message=(
                        f"Failed to remove Docker image at {docker_image_path}. "
                        f"Error: {e}. "
                        "Please check the file permissions and ensure the file is not in use."
                    ),
                )

        ucc_test_dir = os.path.join(self.install_path, self.SUBDIR_PATH)
        if os.path.isdir(ucc_test_dir) and not os.listdir(ucc_test_dir):
            try:
                shutil.rmtree(ucc_test_dir)
            except OSError as e:
                return InstallStatusResult(
                    success=False,
                    message=(
                        f"Failed to remove ucc-test directory at {ucc_test_dir}. "
                        f"Error: {e}. "
                        "Please check the directory permissions and ensure it is not in use."
                    ),
                )

        return InstallStatusResult(success=True)
