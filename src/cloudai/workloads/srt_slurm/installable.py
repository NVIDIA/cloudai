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

import os
import shlex
import subprocess
from pathlib import Path
from typing import cast

import yaml

from cloudai.core import BaseInstaller, InstallStatusResult, PythonExecutable
from cloudai.systems.slurm import SlurmSystem


class SRTSlurmExecutable(PythonExecutable):
    """Install srtctl and the binaries required on the compute nodes."""

    def _runtime_status(self, installer: BaseInstaller) -> InstallStatusResult:
        repo_path = self.git_repo.installed_path or installer.system.install_path / self.git_repo.repo_name
        binaries = ["configs/nats-server", "configs/etcd", "bin/uv"]
        try:
            # Older srt-slurm revisions did not install a Tachometer scraper.
            if "tachometer-scraper-download" in (repo_path / "Makefile").read_text(encoding="utf-8"):
                binaries.append("bin/tachometer-scraper")
            invalid = [
                name
                for name in binaries
                if not (repo_path / name).is_file() or not os.access(repo_path / name, os.X_OK)
            ]
        except OSError as error:
            return InstallStatusResult(False, f"Unable to check srt-slurm runtime: {error}")
        if invalid:
            return InstallStatusResult(False, f"Missing or non-executable runtime binaries: {', '.join(invalid)}")
        return InstallStatusResult(True, "srt-slurm runtime binaries installed")

    def is_installed(self, installer: BaseInstaller) -> InstallStatusResult:
        for result in (self.git_repo.is_installed(installer), super().is_installed(installer)):
            if not result.success:
                return result
        if self.venv_path is None or not os.access(self.venv_path / "bin" / "srtctl", os.X_OK):
            return InstallStatusResult(False, "srtctl executable is missing from the Python environment")
        repo_path = self.git_repo.installed_path or installer.system.install_path / self.git_repo.repo_name
        if not (repo_path / "srtslurm.yaml").is_file():
            return InstallStatusResult(False, "srt-slurm setup configuration is missing")
        return self._runtime_status(installer)

    def _prepare_cluster_config(self, installer: BaseInstaller, repo_path: Path) -> None:
        # Create a placeholder srtslurm.yaml to avoid the interactive prompt during `make setup`.
        # The actual config is supplied by Tests.cmd_args.cluster_config in the CloudAI scenario.
        config = repo_path / "srtslurm.yaml"
        if not config.is_file():
            self._write_cluster_config(installer, config, repo_path)

    def _write_cluster_config(self, installer: BaseInstaller, config_path: Path, repo_path: Path) -> None:
        system = cast(SlurmSystem, installer.system)
        config = {
            "default_account": system.account,
            "default_partition": system.default_partition,
            "gpus_per_node": system.gpus_per_node,
            "srtctl_root": str(repo_path.resolve()),
        }
        config_path.write_text(
            yaml.safe_dump({key: value for key, value in config.items() if value is not None}, sort_keys=False),
            encoding="utf-8",
        )

    def install(self, installer: BaseInstaller) -> InstallStatusResult:
        result = super().install(installer)
        if not result.success:
            return result
        repo_path = self.git_repo.installed_path or installer.system.install_path / self.git_repo.repo_name
        try:
            self._prepare_cluster_config(installer, repo_path)
            if self.is_installed(installer).success:
                return InstallStatusResult(True, "srtctl and runtime binaries already installed")
            system = cast(SlurmSystem, installer.system)
            command = ["srun", "--nodes=1", "--ntasks=1", "--time=00:15:00", f"--partition={system.default_partition}"]
            if system.account:
                command.append(f"--account={system.account}")
            command.extend(shlex.split(system.extra_srun_args or ""))
            command.extend(["make", "setup"])
            completed = subprocess.run(
                command,
                cwd=repo_path,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            return InstallStatusResult(False, f"srt-slurm setup failed: {error}")
        if completed.returncode != 0:
            return InstallStatusResult(False, f"srt-slurm setup failed:\n{completed.stdout}\n{completed.stderr}")
        return self.is_installed(installer)
