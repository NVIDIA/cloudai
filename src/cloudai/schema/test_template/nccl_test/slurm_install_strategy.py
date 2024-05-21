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
from typing import Any, Dict

from cloudai.schema.core import System
from cloudai.schema.core.strategy import InstallStrategy, StrategyRegistry
from cloudai.schema.system import SlurmSystem
from cloudai.schema.system.slurm.strategy import SlurmInstallStrategy
from cloudai.util import CommandShell

from .template import NcclTest


@StrategyRegistry.strategy(InstallStrategy, [SlurmSystem], [NcclTest])
class NcclTestSlurmInstallStrategy(SlurmInstallStrategy):
    """
    Install strategy for NCCL tests on Slurm systems.

    Attributes
        SUBDIR_PATH (str): Subdirectory path where Docker images are stored.
        DOCKER_IMAGE_FILENAME (str): Name of the Docker image file.
    """

    SUBDIR_PATH = "nccl-test"
    DOCKER_IMAGE_FILENAME = "nccl_test.sqsh"

    def __init__(
        self,
        system: System,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        super().__init__(system, env_vars, cmd_args)

    def is_installed(self) -> bool:
        docker_image_path = os.path.join(self.install_path, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME)
        return os.path.isfile(docker_image_path)

    def install(self) -> None:
        if self.is_installed():
            return

        docker_image_dir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        os.makedirs(docker_image_dir_path, exist_ok=True)
        docker_image_path = os.path.join(docker_image_dir_path, self.DOCKER_IMAGE_FILENAME)

        # Remove existing Docker image if it exists
        shell = CommandShell()
        remove_cmd = f"rm -f {docker_image_path}"
        process = shell.execute(remove_cmd)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Failed to remove existing Docker image: {stderr}")

        # Import new Docker image using enroot
        docker_image_url_info = self.cmd_args.get("docker_image_url")
        docker_image_url = docker_image_url_info.get("default") if docker_image_url_info else None
        if docker_image_url is None:
            raise ValueError("docker_image_url not specified or default value " "is None in command-line arguments.")

        enroot_import_cmd = (
            f"srun"
            f" --export=ALL"
            f" --partition={self.slurm_system.default_partition}"
            f" enroot import -o {docker_image_path} docker://{docker_image_url}"
        )

        try:
            subprocess.run(enroot_import_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to import Docker image: {e}") from e

    def uninstall(self) -> None:
        docker_image_path = os.path.join(self.install_path, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME)

        if os.path.isfile(docker_image_path):
            try:
                os.remove(docker_image_path)
            except OSError as e:
                raise OSError(f"Failed to remove Docker image: {e}") from e

        nccl_test_dir = os.path.join(self.install_path, self.SUBDIR_PATH)
        if os.path.isdir(nccl_test_dir) and not os.listdir(nccl_test_dir):
            try:
                shutil.rmtree(nccl_test_dir)
            except OSError as e:
                raise OSError(f"Failed to remove nccl-test directory: {e}") from e
