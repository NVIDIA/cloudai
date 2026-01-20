# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from shutil import rmtree

from cloudai.core import (
    BaseInstaller,
    DockerImage,
    File,
    GitRepo,
    HFModel,
    Installable,
    InstallStatusResult,
    PythonExecutable,
    System,
)
from cloudai.util.hf_model_manager import HFModelManager
from cloudai.util.lazy_imports import lazy


class KubernetesInstaller(BaseInstaller):
    """Installer for Kubernetes systems."""

    def __init__(self, system: System) -> None:
        super().__init__(system)
        self.hf_model_manager = HFModelManager(system.hf_home_path)

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries and Kubernetes configurations.

        This ensures the system environment is properly set up before proceeding with the installation
        or uninstallation processes.

        Returns
            InstallStatusResult: Result containing the status of the prerequisite check and any error message.
        """
        # Check base prerequisites using the parent class method
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            logging.error(f"Prerequisite check failed in base installer: {base_prerequisites_result.message}")
            return InstallStatusResult(False, base_prerequisites_result.message)

        # Load Kubernetes configuration
        try:
            lazy.k8s.config.load_kube_config()
        except Exception as e:
            message = (
                f"Installation failed during prerequisite checking stage because Kubernetes configuration could not "
                f"be loaded. Please ensure that your Kubernetes configuration is properly set up. Original error: {e!r}"
            )
            logging.error(message)
            return InstallStatusResult(False, message)

        logging.info("All prerequisites are met. Proceeding with installation.")
        return InstallStatusResult(True)

    def install_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} installed")
        elif isinstance(item, GitRepo):
            return self._install_one_git_repo(item)
        elif isinstance(item, PythonExecutable):
            return self._install_python_executable(item)
        elif isinstance(item, File):
            item.installed_path = self.system.install_path / item.src.name
            shutil.copyfile(item.src, item.installed_path, follow_symlinks=False)
            return InstallStatusResult(True)
        elif isinstance(item, HFModel):
            return self.hf_model_manager.download_model(item)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} uninstalled")
        elif isinstance(item, GitRepo):
            return self._uninstall_git_repo(item)
        elif isinstance(item, PythonExecutable):
            return self._uninstall_python_executable(item)
        elif isinstance(item, File):
            if item.installed_path != item.src:
                item.installed_path.unlink()
                item._installed_path = None
                return InstallStatusResult(True)
            logging.debug(f"File {item.installed_path} does not exist.")
            return InstallStatusResult(True)
        elif isinstance(item, HFModel):
            return self.hf_model_manager.remove_model(item)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} is installed")
        elif isinstance(item, GitRepo):
            repo_path = self.system.install_path / item.repo_name
            if repo_path.exists():
                item.installed_path = repo_path
                return InstallStatusResult(True)
            return InstallStatusResult(False, f"Git repository {item.url} not cloned")
        elif isinstance(item, PythonExecutable):
            return self._is_python_executable_installed(item)
        elif isinstance(item, File):
            if (self.system.install_path / item.src.name).exists() and (
                self.system.install_path / item.src.name
            ).read_text() == item.src.read_text():
                item.installed_path = self.system.install_path / item.src.name
                return InstallStatusResult(True)
            return InstallStatusResult(False, f"File {self.system.install_path / item.src.name} does not exist")
        elif isinstance(item, HFModel):
            return self.hf_model_manager.is_model_downloaded(item)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} marked as installed")
        elif isinstance(item, GitRepo):
            item.installed_path = self.system.install_path / item.repo_name
            return InstallStatusResult(True)
        elif isinstance(item, PythonExecutable):
            item.git_repo.installed_path = self.system.install_path / item.git_repo.repo_name
            item.venv_path = self.system.install_path / item.venv_name
            return InstallStatusResult(True)
        elif isinstance(item, File):
            item.installed_path = self.system.install_path / item.src.name
            return InstallStatusResult(True)
        elif isinstance(item, HFModel):
            item.installed_path = self.system.hf_home_path  # fake path is OK here as the whole HF home will be mounted
            return InstallStatusResult(True)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def _install_one_git_repo(self, item: GitRepo) -> InstallStatusResult:
        repo_path = self.system.install_path / item.repo_name
        if repo_path.exists():
            item.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_repository(item.url, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(item.commit, repo_path)
        if not res.success:
            return res

        item.installed_path = repo_path
        return InstallStatusResult(True)

    def _install_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._install_one_git_repo(item.git_repo)
        if not res.success:
            return res

        res = self._create_venv(item)
        if not res.success:
            return res

        return InstallStatusResult(True)

    def _install_dependencies(self, item: PythonExecutable) -> InstallStatusResult:
        venv_path = self.system.install_path / item.venv_name

        if not item.git_repo.installed_path:
            return InstallStatusResult(False, "Git repository must be installed before creating virtual environment.")

        project_dir = item.git_repo.installed_path

        if item.project_subpath:
            project_dir = project_dir / item.project_subpath

        pyproject_toml = project_dir / "pyproject.toml"
        requirements_txt = project_dir / "requirements.txt"

        if pyproject_toml.exists() and requirements_txt.exists():
            if item.dependencies_from_pyproject:
                res = self._install_pyproject(venv_path, project_dir)
            else:
                res = self._install_requirements(venv_path, requirements_txt)
        elif pyproject_toml.exists():
            res = self._install_pyproject(venv_path, project_dir)
        elif requirements_txt.exists():
            res = self._install_requirements(venv_path, requirements_txt)
        else:
            return InstallStatusResult(False, "No pyproject.toml or requirements.txt found for installation.")

        return res

    def _clone_repository(self, git_url: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {git_url} into {path}")
        clone_cmd = ["git", "clone"]

        if self.is_low_thread_environment:
            clone_cmd.extend(["-c", "pack.threads=4"])

        clone_cmd.extend([git_url, str(path)])

        logging.debug(f"Running git clone command: {' '.join(clone_cmd)}")
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

    def _create_venv(self, item: PythonExecutable) -> InstallStatusResult:
        venv_path = self.system.install_path / item.venv_name
        logging.debug(f"Creating virtual environment in {venv_path}")
        if venv_path.exists():
            msg = f"Virtual environment already exists at {venv_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        cmd = ["python", "-m", "venv", str(venv_path)]
        logging.debug(f"Creating venv using cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.debug(f"venv creation STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        if result.returncode != 0:
            if venv_path.exists():
                rmtree(venv_path)
            return InstallStatusResult(
                False, f"Failed to create venv:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        res = self._install_dependencies(item)
        if not res.success:
            if venv_path.exists():
                rmtree(venv_path)
            return res

        item.venv_path = self.system.install_path / item.venv_name

        return InstallStatusResult(True)

    def _install_pyproject(self, venv_dir: Path, project_dir: Path) -> InstallStatusResult:
        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", str(project_dir)]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install {project_dir} using pip: {result.stderr}")

        return InstallStatusResult(True)

    def _install_requirements(self, venv_dir: Path, requirements_txt: Path) -> InstallStatusResult:
        if not requirements_txt.is_file():
            return InstallStatusResult(False, f"Requirements file is invalid or does not exist: {requirements_txt}")

        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_txt)]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install dependencies from requirements.txt: {result.stderr}")

        return InstallStatusResult(True)

    def _uninstall_git_repo(self, item: GitRepo) -> InstallStatusResult:
        logging.debug(f"Uninstalling git repository at {item.installed_path=}")
        repo_path = item.installed_path if item.installed_path else self.system.install_path / item.repo_name
        if not repo_path.exists():
            msg = f"Repository {item.url} is not cloned."
            return InstallStatusResult(True, msg)

        logging.debug(f"Removing folder {repo_path}")
        shutil.rmtree(repo_path)
        item.installed_path = None

        return InstallStatusResult(True)

    def _uninstall_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._uninstall_git_repo(item.git_repo)
        if not res.success:
            return res

        logging.debug(f"Uninstalling virtual environment at {item.venv_path=}")
        venv_path = item.venv_path if item.venv_path else self.system.install_path / item.venv_name
        if not venv_path.exists():
            msg = f"Virtual environment {item.venv_name} is not created."
            return InstallStatusResult(True, msg)

        logging.debug(f"Removing folder {venv_path}")
        rmtree(venv_path)
        item.venv_path = None

        return InstallStatusResult(True)

    def _is_python_executable_installed(self, item: PythonExecutable) -> InstallStatusResult:
        repo_path = (
            item.git_repo.installed_path
            if item.git_repo.installed_path
            else self.system.install_path / item.git_repo.repo_name
        )
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {item.git_repo.url} not cloned")
        item.git_repo.installed_path = repo_path

        venv_path = item.venv_path if item.venv_path else self.system.install_path / item.venv_name
        if not venv_path.exists():
            return InstallStatusResult(False, f"Virtual environment not created for {item.git_repo.url}")
        item.venv_path = venv_path

        return InstallStatusResult(True, "Python executable installed")
