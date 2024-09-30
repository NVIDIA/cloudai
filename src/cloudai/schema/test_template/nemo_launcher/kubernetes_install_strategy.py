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

from cloudai import InstallStatusResult, InstallStrategy, System


class NeMoLauncherKubernetesInstallStrategy(InstallStrategy):
    """
    Install strategy for NeMo-Launcher on Kubernetes systems.

    Attributes
        SUBDIR_PATH (str): Subdirectory within the system's install path where the NeMo-Launcher will be stored.
        REPOSITORY_NAME (str): Name of the NeMo-Launcher repository.
        repository_url (str): URL to the NeMo-Launcher Git repository.
        repository_commit_hash (str): Specific commit hash to checkout after cloning the repository.
    """

    SUBDIR_PATH: str = "NeMo-Launcher"
    REPOSITORY_NAME: str = "NeMo-Launcher"

    def __init__(self, system: System, cmd_args: Dict[str, Any]) -> None:
        """
        Initialize the install strategy for NeMo-Launcher on Kubernetes systems.

        Args:
            system (System): The system where NeMo-Launcher will be installed.
            cmd_args (Dict[str, Any]): Command-line arguments including repository URL and commit hash.
        """
        super().__init__(system, cmd_args)
        self.system = system
        self.repository_url: str = cmd_args["repository_url"]
        self.repository_commit_hash: str = cmd_args["repository_commit_hash"]

    def is_installed(self) -> InstallStatusResult:
        """
        Check if NeMo-Launcher is installed on the system.

        Returns
            InstallStatusResult: Status indicating whether NeMo-Launcher is installed.
        """
        subdir_path = self.system.install_path / self.SUBDIR_PATH
        repo_path = subdir_path / self.REPOSITORY_NAME
        repo_installed = repo_path.is_dir()

        if repo_installed:
            return InstallStatusResult(success=True)

        missing_components = [
            f"Repository at {repo_path} from {self.repository_url} " f"with commit hash {self.repository_commit_hash}"
        ]
        return InstallStatusResult(
            success=False,
            message="The following components are missing:\n"
            + "\n".join(f"    - {item}" for item in missing_components),
        )

    def install(self) -> InstallStatusResult:
        """
        Installs NeMo-Launcher by cloning the repository and installing requirements.

        Returns
            InstallStatusResult: Status indicating whether the installation succeeded.
        """
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True, message="NeMo-Launcher is already installed.")

        subdir_path = self.system.install_path / self.SUBDIR_PATH
        subdir_path.mkdir(parents=True, exist_ok=True)

        try:
            self._clone_repository(subdir_path)
            self._install_requirements(subdir_path)
        except RuntimeError as e:
            return InstallStatusResult(success=False, message=str(e))

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        """
        Uninstalls the NeMo-Launcher by removing its cloned repository.

        Returns
            InstallStatusResult: Status indicating whether the uninstallation succeeded.
        """
        subdir_path = self.system.install_path / self.SUBDIR_PATH
        repo_path = subdir_path / self.REPOSITORY_NAME

        if repo_path.exists():
            logging.debug("Removing cloned repository at %s", repo_path)
            try:
                subprocess.run(["rm", "-rf", str(repo_path)], check=True)
            except subprocess.CalledProcessError as e:
                return InstallStatusResult(
                    success=False,
                    message=f"Failed to remove cloned repository. Command: 'rm -rf {repo_path}'. Error: {str(e)}.",
                )
        else:
            logging.warning("Repository does not exist at %s", repo_path)

        return InstallStatusResult(success=True, message="NeMo-Launcher repository removed successfully.")

    def _clone_repository(self, subdir_path: Path) -> None:
        """
        Clones the NeMo-Launcher repository into the specified subdirectory path.

        Args:
            subdir_path (Path): Subdirectory path for installation.

        Raises:
            RuntimeError: If cloning or checking out the commit fails.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME

        if repo_path.exists():
            logging.warning("Repository already exists at %s, clone skipped", repo_path)
        else:
            logging.debug("Cloning NeMo-Launcher repository into %s", repo_path)
            clone_cmd = ["git", "clone", self.repository_url, str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to clone NeMo-Launcher repository. Command: {' '.join(clone_cmd)}. "
                    f"Error: {result.stderr}."
                )

        logging.debug("Checking out specific commit %s in repository", self.repository_commit_hash)
        checkout_cmd = ["git", "checkout", self.repository_commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(repo_path), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout commit {self.repository_commit_hash}. Command: {' '.join(checkout_cmd)}. "
                f"Error: {result.stderr}."
            )

    def _install_requirements(self, subdir_path: Path) -> None:
        """
        Installs the required Python packages from the requirements.txt file in the cloned repository.

        Args:
            subdir_path (Path): Subdirectory path for installation.

        Raises:
            RuntimeError: If installing requirements fails.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME
        requirements_file = repo_path / "requirements.txt"

        if requirements_file.is_file():
            logging.debug("Installing requirements from %s", requirements_file)
            install_cmd = ["pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install requirements for NeMo-Launcher. Command: {' '.join(install_cmd)}. "
                    f"Error: {result.stderr}."
                )
        else:
            logging.warning("requirements.txt not found in %s", repo_path)
