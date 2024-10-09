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

import os
import subprocess
from typing import Iterable

from cloudai import BaseInstaller, InstallStatusResult, TestRun
from cloudai.systems import SlurmSystem


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

    def install(self, trs: Iterable[TestRun]) -> InstallStatusResult:
        """
        Check if the necessary components are installed and install them if not.

        Requires the installation path to be set.

        Args:
            trs (Iterable[TestRun]): Tests to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        if self.install_path is None:
            return InstallStatusResult(
                False, "Installation path is not set. Please set the install path in the system schema."
            )

        prerequisites_result = self._check_prerequisites()
        if not prerequisites_result.success:
            return prerequisites_result

        try:
            self.install_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to create installation directory at {self.install_path}: {e}")

        if not self.install_path.is_dir() or not os.access(self.install_path, os.W_OK):
            return InstallStatusResult(False, f"The installation path {self.install_path} is not writable.")

        return super().install(trs)
