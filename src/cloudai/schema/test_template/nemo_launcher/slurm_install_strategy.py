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
import subprocess
from pathlib import Path
from typing import Any, Dict

from cloudai import InstallStatusResult
from cloudai.systems.slurm.slurm_system import SlurmSystem


class NeMoLauncherSlurmInstallStrategy:
    """
    Install strategy for NeMo-Launcher on Slurm systems.

    Attributes
        SUBDIR_PATH (str): Subdirectory within the system's install path where the NeMo-Launcher and its Docker image
            will be stored.
        REPOSITORY_NAME (str): Name of the NeMo-Launcher repository.
        DOCKER_IMAGE_FILENAME (str): Filename of the Docker image to be downloaded.
        repository_url (str): URL to the NeMo-Launcher Git repository.
        repository_commit_hash (str): Specific commit hash to checkout after cloning the repository.
        docker_image_url (str): URL to the Docker image in a remote container registry.
    """

    SUBDIR_PATH = "NeMo-Launcher"
    REPOSITORY_NAME = "NeMo-Launcher"
    DOCKER_IMAGE_FILENAME = "nemo_launcher.sqsh"

    def __init__(self, system: SlurmSystem, cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, cmd_args)
        self.repository_url = cmd_args["repository_url"]
        self.repository_commit_hash = cmd_args["repository_commit_hash"]
        self.docker_image_url = cmd_args["docker_image_url"]

    def is_installed(self) -> InstallStatusResult:
        subdir_path = self.system.install_path / self.SUBDIR_PATH
        repo_path = subdir_path / self.REPOSITORY_NAME
        repo_installed = repo_path.is_dir()

        docker_image_installed = self.docker_image_cache_manager.check_docker_image_exists(
            self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        ).success

        if repo_installed and docker_image_installed:
            return InstallStatusResult(success=True)
        else:
            missing_components = []
            if not repo_installed:
                missing_components.append(
                    f"Repository at {repo_path} from {self.repository_url} "
                    f"with commit hash {self.repository_commit_hash}"
                )
            if not docker_image_installed:
                docker_image_path = subdir_path / self.DOCKER_IMAGE_FILENAME
                missing_components.append(f"Docker image at {docker_image_path} from URL {self.docker_image_url}")

            return InstallStatusResult(
                success=False,
                message="The following components are missing:\n"
                + "\n".join(f"    - {item}" for item in missing_components),
            )

    def install(self) -> InstallStatusResult:
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True, message="NeMo-Launcher is already installed.")

        subdir_path = self.system.install_path / self.SUBDIR_PATH
        subdir_path.mkdir(parents=True, exist_ok=True)

        try:
            self._clone_repository(subdir_path)
            self._install_requirements(subdir_path)
            docker_image_result = self.docker_image_cache_manager.ensure_docker_image(
                self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
            )
            if not docker_image_result.success:
                raise RuntimeError(
                    "Failed to download and import the Docker image for NeMo-Launcher. "
                    f"Error: {docker_image_result.message}"
                )
        except RuntimeError as e:
            return InstallStatusResult(success=False, message=str(e))

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        docker_image_result = self.docker_image_cache_manager.uninstall_cached_image(
            self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        )
        if not docker_image_result.success:
            return InstallStatusResult(
                success=False,
                message=(
                    "Failed to remove the Docker image for NeMo-Launcher. " f"Error: {docker_image_result.message}"
                ),
            )

        return InstallStatusResult(success=True)

    def _clone_repository(self, subdir_path: Path) -> None:
        """
        Clones NeMo-Launcher repository into specified path if it does not already exist.

        Args:
            subdir_path (Path): Subdirectory path for installation.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME

        if repo_path.exists():
            logging.warning("Repository already exists at %s, clone skipped", repo_path)
        else:
            logging.debug("Cloning NeMo-Launcher repository into %s", repo_path)
            clone_cmd = ["git", "clone", self.repository_url, str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        logging.debug("Checking out specific commit %s in repository", self.repository_commit_hash)
        checkout_cmd = ["git", "checkout", self.repository_commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(repo_path), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout commit: {result.stderr}")

    def _install_requirements(self, subdir_path: Path) -> None:
        """
        Installs the required Python packages from the requirements.txt file in the cloned repository.

        Args:
            subdir_path (Path): Subdirectory path for installation.
        """
        venv_dir = subdir_path / "nemo-venv"
        logging.debug(f"Creating virtual environment in {venv_dir}")
        result = subprocess.run(["python", "-m", "venv", str(venv_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {result.stderr}")

        repo_path = subdir_path / self.REPOSITORY_NAME
        requirements_file = repo_path / "requirements.txt"

        if requirements_file.is_file():
            install_cmd = [(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)]
            logging.debug(f"Installing requirements from {requirements_file} using command: {install_cmd}")
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install requirements: {result.stderr}")
        else:
            logging.warning(f"requirements.txt not found in {repo_path}")
