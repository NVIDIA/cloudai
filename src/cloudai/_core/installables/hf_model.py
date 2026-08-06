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
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from huggingface_hub.utils.tqdm import disable_progress_bars

from .base import Installable, InstallStatusResult

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


@dataclass
class HFModel(Installable):
    """HuggingFace Model object."""

    model_name: str
    _installed_path: Path | None = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        if self._installed_path:
            return self._installed_path
        return Path("hub") / self.model_name

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    @property
    def cache_dir_name(self) -> str:
        """Return this model's directory name in the Hugging Face Hub cache."""
        return f"models--{self.model_name.replace('/', '--')}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HFModel) and other.model_name == self.model_name

    def __hash__(self) -> int:
        return hash(self.model_name)

    @staticmethod
    def _is_hf_home_accessible(installer: "BaseInstaller") -> bool:
        try:
            parent = installer.system.hf_home_path.resolve().parent
            return parent.exists() and parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)
        except (OSError, RuntimeError):
            return False

    def _assume_available_on_execution_nodes(self, installer: "BaseInstaller", operation: str) -> InstallStatusResult:
        self.installed_path = installer.system.hf_home_path
        return InstallStatusResult(
            True,
            f"HF home path '{installer.system.hf_home_path}' is not accessible locally, "
            f"skipping {operation} of {self.model_name}. "
            "Ensure the model is available on execution nodes.",
        )

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        if not self._is_hf_home_accessible(installer):
            return self._assume_available_on_execution_nodes(installer, "download")

        hf_home_path = installer.system.hf_home_path
        logging.debug(f"Downloading HF model {self.model_name} into {hf_home_path / 'hub'}")
        disable_progress_bars()
        try:
            local_path: str = huggingface_hub.snapshot_download(
                repo_id=self.model_name,
                cache_dir=hf_home_path.absolute() / "hub",
            )
            self.installed_path = Path(local_path)
        except Exception as e:
            return InstallStatusResult(False, f"Failed to download HF model {self.model_name}: {e}")

        return InstallStatusResult(True)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        if not self._is_hf_home_accessible(installer):
            return self._assume_available_on_execution_nodes(installer, "removal")

        hf_home_path = installer.system.hf_home_path
        logging.debug(f"Removing HF model {self.model_name} from {hf_home_path / 'hub'}")
        res = self.is_installed(installer)
        if not res.success:
            return InstallStatusResult(True, f"HF model {self.model_name} is not downloaded.")

        cmd = ["hf", "cache", "rm", "-y", f"model/{self.model_name}"]
        env = os.environ | {"HF_HOME": str(hf_home_path.absolute())}
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        logging.debug(
            f"Run {cmd=} with HF_HOME={env['HF_HOME']} returned code {p.returncode}, "
            f"stdout: {p.stdout}, stderr: {p.stderr}"
        )
        if p.returncode != 0:
            return InstallStatusResult(False, f"Failed to remove HF model {self.model_name}: {p.stderr} {p.stdout}")

        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        if not self._is_hf_home_accessible(installer):
            self.installed_path = installer.system.hf_home_path
            return InstallStatusResult(True)

        hf_home_path = installer.system.hf_home_path
        logging.debug(f"Checking if HF model {self.model_name} is already downloaded in {hf_home_path / 'hub'}")
        disable_progress_bars()
        try:
            local_path: str = huggingface_hub.snapshot_download(
                repo_id=self.model_name,
                cache_dir=hf_home_path.absolute() / "hub",
                local_files_only=True,
            )
            self.installed_path = Path(local_path)
        except Exception as e:
            return InstallStatusResult(False, f"HF model {self.model_name} is not available locally: {e}")

        return InstallStatusResult(True, local_path)

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.hf_home_path
        return InstallStatusResult(True)
