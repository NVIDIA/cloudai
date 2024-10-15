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
from shutil import rmtree

from cloudai import BaseInstaller, InstallStatusResult
from cloudai.installer.installables import DockerImage, GitRepo, Installable, PythonExecutable
from cloudai.systems import SlurmSystem
from cloudai.util.docker_image_cache_manager import DockerImageCacheManager, DockerImageCacheResult


class SlurmInstaller(BaseInstaller):
    """
    Installer for systems that use the Slurm scheduler.

    Handles the installation of benchmarks or test templates for Slurm-managed systems.

    Attributes
        PREREQUISITES (List[str]): A list of required binaries for the installer.
        REQUIRED_SRUN_OPTIONS (List[str]): A list of required srun options to check.
        install_path (Path): Path where the benchmarks are to be installed. This is optional since uninstallation does
            not require it.
    """

    PREREQUISITES = ["git", "sbatch", "sinfo", "squeue", "srun", "scancel"]
    REQUIRED_SRUN_OPTIONS = [
        "--mpi",
        "--gpus-per-node",
        "--ntasks-per-node",
        "--container-image",
        "--container-mounts",
    ]

    def __init__(self, system: SlurmSystem):
        """
        Initialize the SlurmInstaller with a system object and an optional installation path.

        Args:
            system (SlurmSystem): The system schema object.
        """
        super().__init__(system)
        self.system = system
        self.install_path = self.system.install_path
        self.docker_image_cache_manager = DockerImageCacheManager(
            self.system.install_path, self.system.cache_docker_images_locally, self.system.default_partition
        )

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries and specific srun options, raising an error if any are missing.

        This ensures the system environment is properly set up before proceeding with the installation or uninstallation
        processes.

        Returns
            InstallStatusResult: Result containing the status and any error message.
        """
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            return InstallStatusResult(False, base_prerequisites_result.message)

        try:
            self._check_required_binaries()
            self._check_srun_options()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        """Check for the presence of required binaries, raising an error if any are missing."""
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

    def _check_srun_options(self) -> None:
        """
        Check for the presence of specific srun options.

        Calls `srun --help` and verifying the options. Raises an exception if any required options are missing.
        """
        try:
            result = subprocess.run(["srun", "--help"], text=True, capture_output=True, check=True)
            help_output = result.stdout
        except subprocess.CalledProcessError as e:
            raise EnvironmentError(f"Failed to execute 'srun --help': {e}") from e

        missing_options = [option for option in self.REQUIRED_SRUN_OPTIONS if option not in help_output]
        if missing_options:
            missing_options_str = ", ".join(missing_options)
            raise EnvironmentError(f"Required srun options missing: {missing_options_str}")

    def install_one(self, item: Installable) -> InstallStatusResult:
        """
        Install a single item.

        Args:
            item (Installable): The item to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        logging.debug(f"Attempt to install {item}")
        if isinstance(item, DockerImage):
            res = self._install_docker_image(item)
            return InstallStatusResult(res.success, res.message)
        elif isinstance(item, PythonExecutable):
            return self._install_python_executable(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        """
        Uninstall a single item.

        Args:
            item (Installable): The item to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        logging.info(f"Attempt to uninstall {item}")
        if isinstance(item, DockerImage):
            res = self._uninstall_docker_image(item)
            return InstallStatusResult(res.success, res.message)
        elif isinstance(item, PythonExecutable):
            return self._uninstall_python_executable(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            res = self.docker_image_cache_manager.check_docker_image_exists(item.url, item.cache_filename)
            if res.success and res.docker_image_path:
                item.installed_path = res.docker_image_path
            return InstallStatusResult(res.success, res.message)
        elif isinstance(item, PythonExecutable):
            return self._is_python_executable_installed(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def _install_docker_image(self, item: DockerImage) -> DockerImageCacheResult:
        res = self.docker_image_cache_manager.ensure_docker_image(item.url, item.cache_filename)
        if res.success and res.docker_image_path:
            item.installed_path = res.docker_image_path
        return res

    def _uninstall_docker_image(self, item: DockerImage) -> DockerImageCacheResult:
        res = self.docker_image_cache_manager.uninstall_cached_image(item.cache_filename)
        if res.success:
            item.installed_path = item.url
        return res

    def _install_one_git_repo(self, item: GitRepo) -> InstallStatusResult:
        repo_path = self.install_path / item.repo_name
        if repo_path.exists():
            item.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_repository(item.git_url, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(item.commit_hash, repo_path)
        if not res.success:
            return res

        item.installed_path = repo_path
        return InstallStatusResult(True)

    def _install_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._install_one_git_repo(item.git_repo)
        if not res.success:
            return res

        venv_path = self.install_path / item.venv_name
        res = self._create_venv(venv_path)
        if not res.success:
            return res

        requirements_txt = item.git_repo.installed_path / "requirements.txt"
        res = self._install_requirements(venv_path, requirements_txt)
        if not res.success:
            return res

        item.venv_path = venv_path

        return InstallStatusResult(True)

    def _clone_repository(self, git_url: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {git_url} into {path}")
        clone_cmd = ["git", "clone", git_url, str(path)]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to clone repository: {result.stderr}")
        return InstallStatusResult(True)

    def _checkout_commit(self, commit_hash: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Checking out specific commit in {path}: {commit_hash}")
        checkout_cmd = ["git", "checkout", commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(path), capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to checkout commit: {result.stderr}")
        return InstallStatusResult(True)

    def _create_venv(self, venv_dir: Path) -> InstallStatusResult:
        logging.debug(f"Creating virtual environment in {venv_dir}")
        if venv_dir.exists():
            msg = f"Virtual environment already exists at {venv_dir}."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        result = subprocess.run(["python", "-m", "venv", str(venv_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to create venv: {result.stderr}")
        return InstallStatusResult(True)

    def _install_requirements(self, venv_dir: Path, requirements_txt: Path) -> InstallStatusResult:
        if not requirements_txt.is_file() or not requirements_txt.exists():
            msg = f"Requirements file is invalid or does not exist: {requirements_txt}"
            logging.warning(msg)
            return InstallStatusResult(False, msg)

        install_cmd = [(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_txt)]
        logging.debug(f"Installing requirements from {requirements_txt} using command: {install_cmd}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install requirements: {result.stderr}")

        return InstallStatusResult(True)

    def _uninstall_git_repo(self, item: GitRepo) -> InstallStatusResult:
        if not item.installed_path.exists():
            msg = f"Repository {item.git_url} is not cloned."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        rmtree(item.installed_path)
        item._installed_path = None

        return InstallStatusResult(True)

    def _uninstall_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._uninstall_git_repo(item.git_repo)
        if not res.success:
            return res

        if not item.venv_path.exists():
            msg = f"Virtual environment {item.venv_name} is not created."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        rmtree(item.venv_path)
        item._venv_path = None

        return InstallStatusResult(True)

    def _is_python_executable_installed(self, item: PythonExecutable) -> InstallStatusResult:
        repo_path = self.install_path / item.git_repo.installed_path
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {item.git_repo.git_url} not cloned")
        item.git_repo.installed_path = repo_path

        venv_path = self.install_path / item.venv_path
        if not venv_path.exists():
            return InstallStatusResult(False, f"Virtual environment not created for {item.git_repo.git_url}")
        item.venv_path = venv_path

        return InstallStatusResult(True, "Python executable installed")
