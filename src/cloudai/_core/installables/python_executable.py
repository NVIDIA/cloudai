# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import uv

from .base import Installable, InstallStatusResult
from .git_repo import GitRepo

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


_PYTHON_VERSION_MARKER = ".cloudai-python-version"


@dataclass
class PythonExecutable(Installable):
    """Python executable object."""

    git_repo: GitRepo
    venv_path: Optional[Path] = None
    project_subpath: Optional[Path] = None
    dependencies_from_pyproject: bool = True

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return (
            isinstance(other, PythonExecutable)
            and other.git_repo.url == self.git_repo.url
            and other.git_repo.commit == self.git_repo.commit
        )

    def __hash__(self) -> int:
        """Hash the installable object."""
        return self.git_repo.__hash__()

    def __str__(self) -> str:
        """Return the string representation of the python executable."""
        return f"PythonExecutable(git_url={self.git_repo.url}, commit_hash={self.git_repo.commit})"

    @property
    def venv_name(self) -> str:
        return f"{self.git_repo.repo_name}-venv"

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        res = self.git_repo.install(installer)
        if not res.success:
            return res

        return self._create_venv(installer)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        res = self.git_repo.uninstall(installer)
        if not res.success:
            return res

        logging.debug(f"Uninstalling virtual environment at {self.venv_path=}")
        venv_path = self.venv_path if self.venv_path else installer.system.install_path / self.venv_name
        if not venv_path.exists():
            return InstallStatusResult(True, f"Virtual environment {self.venv_name} is not created.")

        logging.debug(f"Removing folder {venv_path}")
        shutil.rmtree(venv_path)
        self.venv_path = None

        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = (
            self.git_repo.installed_path
            if self.git_repo.installed_path
            else installer.system.install_path / self.git_repo.repo_name
        )
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {self.git_repo.url} not cloned")
        self.git_repo.installed_path = repo_path

        venv_path = self.venv_path if self.venv_path else installer.system.install_path / self.venv_name
        if not venv_path.exists():
            return InstallStatusResult(False, f"Virtual environment not created for {self.git_repo.url}")

        python_version = self._resolve_python_version(repo_path)
        if not self._python_version_matches(venv_path, python_version):
            return InstallStatusResult(False, f"Virtual environment uses a different Python than {python_version}")

        self.venv_path = venv_path

        return InstallStatusResult(True, "Python executable installed")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.git_repo.installed_path = installer.system.install_path / self.git_repo.repo_name
        self.venv_path = installer.system.install_path / self.venv_name
        return InstallStatusResult(True)

    def _create_venv(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = installer.system.install_path / self.venv_name
        project_dir = typing.cast(Path, self.git_repo.installed_path)
        python_version = self._resolve_python_version(project_dir)
        logging.debug(f"Creating virtual environment in {venv_path}")
        if venv_path.exists() and self._python_version_matches(venv_path, python_version):
            msg = f"Virtual environment already exists at {venv_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        project_dir = project_dir / self.project_subpath if self.project_subpath else project_dir
        if not project_dir.is_dir():
            return InstallStatusResult(False, f"Python project directory does not exist: {project_dir}")

        try:
            uv_bin = uv.find_uv_bin()
        except OSError as e:
            return InstallStatusResult(False, f"Cannot create virtual environment: {e}")

        if venv_path.exists():
            shutil.rmtree(venv_path)

        cmd = [uv_bin, "venv", "--python", python_version, "--seed", str(venv_path)]
        logging.debug(f"Creating venv using cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
        logging.debug(f"venv creation STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        if result.returncode != 0:
            if venv_path.exists():
                shutil.rmtree(venv_path)
            return InstallStatusResult(
                False, f"Failed to create venv using uv:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        res = self._install_dependencies(installer)
        if not res.success:
            if venv_path.exists():
                shutil.rmtree(venv_path)
            return res

        if python_version != sys.executable:
            (venv_path / _PYTHON_VERSION_MARKER).write_text(f"{python_version}\n", encoding="utf-8")

        self.venv_path = installer.system.install_path / self.venv_name

        return InstallStatusResult(True)

    def _resolve_python_version(self, repo_path: Path) -> str:
        if self.git_repo.python_version:
            return self.git_repo.python_version.strip()

        repo_root = repo_path.resolve()
        current = (repo_path / self.project_subpath if self.project_subpath else repo_path).resolve()
        if current != repo_root and repo_root not in current.parents:
            return sys.executable

        while True:
            version_file = current / ".python-version"
            if version_file.is_file():
                python_version = version_file.read_text(encoding="utf-8").strip()
                if python_version:
                    return python_version

            if current == repo_root:
                break
            current = current.parent

        return sys.executable

    @staticmethod
    def _python_version_matches(venv_path: Path, python_version: str) -> bool:
        if python_version == sys.executable:
            return True
        marker = venv_path / _PYTHON_VERSION_MARKER
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == python_version

    def _install_dependencies(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = installer.system.install_path / self.venv_name

        if not self.git_repo.installed_path:
            return InstallStatusResult(False, "Git repository must be installed before creating virtual environment.")

        project_dir = self.git_repo.installed_path

        if self.project_subpath:
            project_dir = project_dir / self.project_subpath

        pyproject_toml = project_dir / "pyproject.toml"
        requirements_txt = project_dir / "requirements.txt"

        if pyproject_toml.exists() and requirements_txt.exists():
            if self.dependencies_from_pyproject:
                return self._install_pyproject(venv_path, project_dir)
            return self._install_requirements(venv_path, requirements_txt)
        if pyproject_toml.exists():
            return self._install_pyproject(venv_path, project_dir)
        if requirements_txt.exists():
            return self._install_requirements(venv_path, requirements_txt)

        return InstallStatusResult(False, "No pyproject.toml or requirements.txt found for installation.")

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
