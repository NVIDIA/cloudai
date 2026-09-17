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

import dataclasses
import hashlib
import logging
import pathlib
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import uv

from .base import Installable, InstallStatusResult

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


@dataclasses.dataclass
class PythonEnvironment(Installable):
    """Python virtual environment installed from package requirements."""

    name: str
    python_version: str
    requirements: list[str] = dataclasses.field(default_factory=list)
    venv_path: pathlib.Path | None = None

    def __eq__(self, other: object) -> bool:
        """Check if two Python environments are equal."""
        return (
            isinstance(other, PythonEnvironment)
            and other.name == self.name
            and other.python_version == self.python_version
            and other.requirements == self.requirements
        )

    def __hash__(self) -> int:
        """Hash the Python environment."""
        return hash((self.name, self.python_version, tuple(self.requirements)))

    @property
    def venv_name(self) -> str:
        requirements_hash = hashlib.sha256("\n".join(self.requirements).encode()).hexdigest()[:12]
        name = self._sanitize(self.name)
        python_version = self._sanitize(self.python_version)
        return f"{name}-py{python_version}-{requirements_hash}"

    def python_path(self, install_path: pathlib.Path) -> pathlib.Path:
        venv_path = self.venv_path or install_path / self.venv_name
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        installed = self.is_installed(installer)
        if installed.success:
            return installed

        try:
            uv_bin = uv.find_uv_bin()
        except OSError as e:
            return InstallStatusResult(False, f"Cannot install Python environment: {e}")

        res = self._ensure_python_version(uv_bin)
        if not res.success:
            return res

        venv_path = installer.system.install_path / self.venv_name
        res = self._create_venv(uv_bin, venv_path)
        if not res.success:
            self._cleanup_venv(venv_path)
            self.venv_path = None
            return res

        res = self._install_requirements(uv_bin, installer)
        if not res.success:
            self._cleanup_venv(venv_path)
            self.venv_path = None
            return res

        self.venv_path = venv_path
        return InstallStatusResult(True, f"Python environment installed at {venv_path}.")

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = self.venv_path or installer.system.install_path / self.venv_name
        if not venv_path.exists():
            self.venv_path = None
            return InstallStatusResult(True, f"Python environment {self.venv_name} is not installed.")

        logging.debug(f"Removing Python environment at {venv_path}")
        shutil.rmtree(venv_path)
        self.venv_path = None
        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = self.venv_path or installer.system.install_path / self.venv_name
        python_path = self.python_path(installer.system.install_path)
        if not venv_path.exists():
            return InstallStatusResult(False, f"Python environment {self.venv_name} is not created.")
        if not python_path.is_file():
            return InstallStatusResult(False, f"Python executable does not exist at {python_path}.")

        self.venv_path = venv_path
        return InstallStatusResult(True, "Python environment installed.")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.venv_path = installer.system.install_path / self.venv_name
        return InstallStatusResult(True)

    def _ensure_python_version(self, uv: str) -> InstallStatusResult:
        cmd = [uv, "python", "install", self.python_version]
        logging.debug(f"Ensuring Python version using: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(
                False, f"Failed to install Python {self.python_version} using uv: {result.stderr}"
            )
        return InstallStatusResult(True)

    def _create_venv(self, uv: str, venv_path: pathlib.Path) -> InstallStatusResult:
        cmd = [uv, "venv", "--python", self.python_version, str(venv_path)]
        logging.debug(f"Creating Python environment using: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to create Python environment using uv: {result.stderr}")
        self.venv_path = venv_path
        return InstallStatusResult(True)

    def _install_requirements(self, uv: str, installer: "BaseInstaller") -> InstallStatusResult:
        if not self.requirements:
            return InstallStatusResult(True)

        cmd = [
            uv,
            "pip",
            "install",
            "--python",
            str(self.python_path(installer.system.install_path)),
            *self.requirements,
        ]
        logging.debug(f"Installing Python requirements using: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install Python requirements using uv: {result.stderr}")
        return InstallStatusResult(True)

    @staticmethod
    def _cleanup_venv(venv_path: pathlib.Path) -> None:
        if venv_path.exists():
            shutil.rmtree(venv_path)

    @staticmethod
    def _sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
