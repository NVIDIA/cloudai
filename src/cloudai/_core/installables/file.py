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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .base import Installable, InstallStatusResult

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


@dataclass
class File(Installable):
    """File object."""

    src: Path
    _installed_path: Optional[Path] = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        return self._installed_path or self.src

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, File) and other.src == self.src

    def __hash__(self) -> int:
        return hash(self.src)

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.src.name
        shutil.copyfile(self.src, self.installed_path, follow_symlinks=False)
        return InstallStatusResult(True)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        if self.installed_path != self.src:
            self.installed_path.unlink()
            self._installed_path = None
            return InstallStatusResult(True)
        logging.debug(f"File {self.installed_path} does not exist.")
        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        installed_path = installer.system.install_path / self.src.name
        if not installed_path.exists():
            return InstallStatusResult(False, f"File {installed_path} does not exist")
        if installed_path.read_bytes() != self.src.read_bytes():
            return InstallStatusResult(False, f"File {installed_path} differs from {self.src}")
        self.installed_path = installed_path
        return InstallStatusResult(True)

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.src.name
        return InstallStatusResult(True)
