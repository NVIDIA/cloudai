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

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ._uv import resolve_uv_bin
from .base import Installable, InstallStatusResult
from .git_repo import GitRepo

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


_PYTHON_REQUEST_MARKER = ".cloudai-python-request"


@dataclass
class PythonExecutable(Installable):
    """Python executable object."""

    git_repo: GitRepo
    venv_path: Optional[Path] = None
    project_subpath: Optional[Path] = None
    dependencies_from_pyproject: bool = True

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, PythonExecutable) and other._identity() == self._identity()

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash(self._identity())

    def __str__(self) -> str:
        """Return the string representation of the python executable."""
        return f"PythonExecutable(git_url={self.git_repo.url}, commit_hash={self.git_repo.commit})"

    @property
    def venv_name(self) -> str:
        base_name = f"{self.git_repo.repo_name}-venv"
        if self._uses_default_environment_config():
            return base_name

        payload = json.dumps(self._identity(), separators=(",", ":"), ensure_ascii=True)
        config_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"{base_name}-{config_hash}"

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

        python_path = self._python_path(venv_path)
        if not python_path.is_file():
            return InstallStatusResult(False, f"Python executable does not exist at {python_path}")

        request_res = self._get_python_request(repo_path)
        if isinstance(request_res, InstallStatusResult):
            return request_res
        python_request, is_pinned = request_res
        if is_pinned and not self._request_marker_matches(venv_path, python_request):
            return InstallStatusResult(
                False,
                f"Python interpreter request marker is missing or does not match {python_request!r}",
            )

        self.venv_path = venv_path
        return InstallStatusResult(True, "Python executable installed")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.git_repo.installed_path = installer.system.install_path / self.git_repo.repo_name
        self.venv_path = installer.system.install_path / self.venv_name
        return InstallStatusResult(True)

    def _create_venv(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = installer.system.install_path / self.venv_name
        repo_path = self.git_repo.installed_path or installer.system.install_path / self.git_repo.repo_name
        project_dir = self._project_dir(repo_path)
        request_res = self._get_python_request(repo_path)
        if isinstance(request_res, InstallStatusResult):
            return request_res
        python_request, is_pinned = request_res

        logging.debug(f"Creating virtual environment in {venv_path}")
        existing_res = self._prepare_existing_venv(venv_path, python_request, is_pinned)
        if existing_res is not None:
            return existing_res

        if not project_dir.is_dir():
            return InstallStatusResult(False, f"Python project directory does not exist: {project_dir}")

        try:
            uv = resolve_uv_bin()
        except RuntimeError as e:
            return InstallStatusResult(False, f"Cannot create virtual environment: {e}")

        cmd = [uv, "venv", "--python", python_request, "--seed", str(venv_path)]
        logging.debug(f"Creating venv using cmd: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
        except OSError as e:
            return self._failure_with_cleanup(venv_path, f"Failed to create venv using uv: {e}")
        logging.debug(f"venv creation STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        if result.returncode != 0:
            return self._failure_with_cleanup(
                venv_path,
                f"Failed to create venv using uv:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

        res = self._install_dependencies(installer)
        if not res.success:
            return self._failure_with_cleanup(venv_path, res.message)

        marker_res = self._write_request_marker(venv_path, python_request, is_pinned)
        if marker_res is not None:
            return marker_res

        self.venv_path = venv_path

        return InstallStatusResult(True)

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
        install_cmd = [str(self._python_path(venv_dir)), "-m", "pip", "install", str(project_dir)]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        try:
            result = subprocess.run(install_cmd, capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to install {project_dir} using pip: {e}")

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install {project_dir} using pip: {result.stderr}")

        return InstallStatusResult(True)

    def _install_requirements(self, venv_dir: Path, requirements_txt: Path) -> InstallStatusResult:
        if not requirements_txt.is_file():
            return InstallStatusResult(False, f"Requirements file is invalid or does not exist: {requirements_txt}")

        install_cmd = [
            str(self._python_path(venv_dir)),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_txt),
        ]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        try:
            result = subprocess.run(install_cmd, capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to install dependencies from requirements.txt: {e}")

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install dependencies from requirements.txt: {result.stderr}")

        return InstallStatusResult(True)

    def _identity(self) -> tuple[str, str, Optional[str], Optional[str], bool]:
        python_version = self.git_repo.python_version
        normalized_python_version = python_version.strip() if python_version is not None else None
        project_subpath = Path(self.project_subpath).as_posix() if self.project_subpath is not None else None
        return (
            self.git_repo.url,
            self.git_repo.commit,
            normalized_python_version,
            project_subpath,
            self.dependencies_from_pyproject,
        )

    def _uses_default_environment_config(self) -> bool:
        return (
            self.git_repo.python_version is None and self.project_subpath is None and self.dependencies_from_pyproject
        )

    def _project_dir(self, repo_path: Path) -> Path:
        return repo_path / self.project_subpath if self.project_subpath is not None else repo_path

    def _resolve_python_request(self, repo_path: Path) -> tuple[str, bool]:
        """Resolve the uv Python request and whether it came from an explicit pin."""
        if self.git_repo.python_version is not None:
            request = self.git_repo.python_version.strip()
            if not request:
                raise ValueError("Git repository python_version must not be empty.")
            return request, True

        repo_root = repo_path.resolve()
        current = self._project_dir(repo_path).resolve()
        try:
            current.relative_to(repo_root)
        except ValueError:
            return sys.executable, False

        while True:
            version_file = current / ".python-version"
            if version_file.is_file():
                try:
                    request = version_file.read_text(encoding="utf-8").strip()
                except OSError as e:
                    raise RuntimeError(f"Failed to read Python version from {version_file}: {e}") from e
                if not request:
                    raise ValueError(f"Python version file is empty: {version_file}")
                return request, True

            if current == repo_root:
                break
            current = current.parent

        return sys.executable, False

    def _get_python_request(self, repo_path: Path) -> tuple[str, bool] | InstallStatusResult:
        try:
            return self._resolve_python_request(repo_path)
        except (OSError, RuntimeError, ValueError) as e:
            return InstallStatusResult(False, f"Failed to resolve Python interpreter request: {e}")

    @staticmethod
    def _python_path(venv_path: Path) -> Path:
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    @staticmethod
    def _request_marker_matches(venv_path: Path, python_request: str) -> bool:
        marker = venv_path / _PYTHON_REQUEST_MARKER
        try:
            return marker.is_file() and marker.read_text(encoding="utf-8").strip() == python_request
        except OSError:
            return False

    def _prepare_existing_venv(
        self, venv_path: Path, python_request: str, is_pinned: bool
    ) -> Optional[InstallStatusResult]:
        if not venv_path.exists():
            return None

        has_python = self._python_path(venv_path).is_file()
        has_matching_request = not is_pinned or self._request_marker_matches(venv_path, python_request)
        if has_python and has_matching_request:
            self.venv_path = venv_path
            msg = f"Virtual environment already exists at {venv_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        logging.info(f"Recreating stale virtual environment at {venv_path}")
        try:
            self._cleanup_venv(venv_path)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to remove stale virtual environment {venv_path}: {e}")
        return None

    @classmethod
    def _write_request_marker(
        cls, venv_path: Path, python_request: str, is_pinned: bool
    ) -> Optional[InstallStatusResult]:
        if not is_pinned:
            return None

        marker = venv_path / _PYTHON_REQUEST_MARKER
        try:
            marker.write_text(f"{python_request}\n", encoding="utf-8")
        except OSError as e:
            return cls._failure_with_cleanup(
                venv_path,
                f"Failed to record Python interpreter request {python_request!r} in {marker}: {e}",
            )
        return None

    @staticmethod
    def _cleanup_venv(venv_path: Path) -> None:
        if venv_path.is_symlink() or venv_path.is_file():
            venv_path.unlink()
        elif venv_path.exists():
            shutil.rmtree(venv_path)

    @classmethod
    def _failure_with_cleanup(cls, venv_path: Path, message: str) -> InstallStatusResult:
        try:
            cls._cleanup_venv(venv_path)
        except OSError as e:
            message = f"{message}\nFailed to clean up partial virtual environment {venv_path}: {e}"
        return InstallStatusResult(False, message)
