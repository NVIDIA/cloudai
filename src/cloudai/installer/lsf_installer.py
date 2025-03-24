# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai import BaseInstaller, DockerImage, File, GitRepo, Installable, InstallStatusResult, PythonExecutable
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.systems import LSFSystem


class LSFInstaller(BaseInstaller):
    """
    Installer for systems that use the LSF scheduler.

    Handles the installation of benchmarks or test templates for LSF-managed systems.

    Attributes:
        PREREQUISITES (List[str]): A list of required binaries for the installer.
        install_path (Path): Path where the benchmarks are to be installed.
    """

    PREREQUISITES = ("bsub", "bjobs", "bhosts", "lsid", "lsload")

    def __init__(self, system: LSFSystem):
        """
        Initialize the LSFInstaller with a system object.

        Args:
            system (LSFSystem): The system schema object.
        """
        super().__init__(system)
        self.system = system

    @property
    def slurm_installer(self) -> SlurmInstaller:
        """
        Lazily initialize and return a SlurmInstaller instance.

        Returns:
            SlurmInstaller: The SlurmInstaller instance.
        """
        if not hasattr(self, "_slurm_installer"):
            from cloudai.systems import SlurmSystem

            if not isinstance(self.system, SlurmSystem):
                raise TypeError("The system must be of type SlurmSystem to use SlurmInstaller.")
            self._slurm_installer = SlurmInstaller(self.system)
        return self._slurm_installer

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries, raising an error if any are missing.

        Returns:
            InstallStatusResult: Result containing the status and any error message.
        """
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            return InstallStatusResult(False, base_prerequisites_result.message)

        try:
            self._check_required_binaries()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        """Check for the presence of required binaries, raising an error if any are missing."""
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

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
            logging.info(f"Skipping installation of Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")
        elif isinstance(item, GitRepo):
            return self.slurm_installer._install_one_git_repo(item)
        elif isinstance(item, PythonExecutable):
            return self.slurm_installer._install_python_executable(item)
        elif isinstance(item, File):
            return self.slurm_installer.install_one(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        """
        Uninstall a single item.

        Args:
            item (Installable): The item to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        logging.debug(f"Attempt to uninstall {item!r}")
        if isinstance(item, PythonExecutable):
            return self.slurm_installer._uninstall_python_executable(item)
        elif isinstance(item, GitRepo):
            return self.slurm_installer._uninstall_git_repo(item)
        elif isinstance(item, File):
            return self.slurm_installer.uninstall_one(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        """
        Check if a single item is installed.

        Args:
            item (Installable): The item to check.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        if isinstance(item, DockerImage):
            logging.info(f"Skipping installation check for Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")
        elif isinstance(item, GitRepo):
            return self.slurm_installer.is_installed_one(item)
        elif isinstance(item, PythonExecutable):
            return self.slurm_installer._is_python_executable_installed(item)
        elif isinstance(item, File):
            return self.slurm_installer.is_installed_one(item)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        """
        Mark a single item as installed.

        Args:
            item (Installable): The item to mark as installed.

        Returns:
            InstallStatusResult: Result containing the status and error message if any.
        """
        logging.debug(f"Marking {item!r} as installed.")

        return self.slurm_installer.mark_as_installed_one(item)
