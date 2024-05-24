# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import os
import shutil
import subprocess
from typing import Iterable, cast

import toml

from cloudai._core.system import System
from cloudai._core.test_template import TestTemplate
from cloudai.schema.system import SlurmSystem

from .base_installer import BaseInstaller
from .installer import Installer


@Installer.register("slurm")
class SlurmInstaller(BaseInstaller):
    """
    Installer for systems that use the Slurm scheduler.

    Handles the installation of benchmarks or test templates for Slurm-managed systems.

    Attributes
        CONFIG_FILE_NAME (str): The name of the configuration file.
        PREREQUISITES (List[str]): A list of required binaries for the installer.
        REQUIRED_SRUN_OPTIONS (List[str]): A list of required srun options to check.
        install_path (str): Path where the benchmarks are to be
            installed. This is optional since uninstallation does not
            require it.
        config_path (str): Path to the installation configuration file.
    """

    CONFIG_FILE_NAME = ".cloudai.toml"
    PREREQUISITES = ["git", "sbatch", "sinfo", "squeue", "srun", "scancel"]
    REQUIRED_SRUN_OPTIONS = [
        "--mpi",
        "--gpus-per-node",
        "--ntasks-per-node",
        "--container-image",
        "--container-mounts",
    ]

    def __init__(self, system: System):
        """
        Initialize the BaseInstaller with a system object and an optional installation path.

        Args:
            system (System): The system schema object.
        """
        super().__init__(system)
        slurm_system = cast(SlurmSystem, self.system)
        self.install_path = slurm_system.install_path
        self.config_path = os.path.join(os.path.expanduser("~"), self.CONFIG_FILE_NAME)

    def _check_prerequisites(self) -> None:
        """
        Check for the presence of required binaries and specific srun options, raising an error if any are missing.

        This ensures the system environment is properly set up before proceeding with the installation or uninstallation
        processes.
        """
        super()._check_prerequisites()
        self._check_required_binaries()
        self._check_srun_options()

    def _check_required_binaries(self) -> None:
        """Check for the presence of required binaries, raising an error if anyare missing."""
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary {binary} is not installed.")

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
            raise EnvironmentError("Required srun options missing: " f"{missing_options_str}")

    def _write_config(self) -> None:
        """Write the installation configuration to a TOML file atomically."""
        absolute_install_path = os.path.abspath(self.install_path)
        config_data = {"install_path": absolute_install_path}

        try:
            with open(self.config_path, "w") as file:
                toml.dump(config_data, file)
        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(self.config_path)
            raise e

    def _read_config(self) -> dict:
        """
        Read the installation configuration from a TOML file.

        Returns
            dict: Configuration, including installation path.
        """
        try:
            with open(self.config_path, "r") as file:
                return toml.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError("Configuration file not found.") from e

    def _remove_config(self) -> None:
        """Remove the installation configuration file."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def is_installed(self, test_templates: Iterable[TestTemplate]) -> bool:
        """
        Check if the necessary components for the provided test templates are already installed.

        Verify the existence of the configuration file and the installation status of each test template.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                check for installation.

        Returns:
            bool: True if the configuration file exists and all test templates
            are installed, False otherwise.
        """
        if not os.path.exists(self.config_path):
            return False

        try:
            self._read_config()
        except FileNotFoundError:
            return False

        return super().is_installed(test_templates)

    def install(self, test_templates: Iterable[TestTemplate]) -> None:
        """
        Check if the necessary components are installed and install them if not.

        Requires the installation path to be set.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                be installed.
        """
        if self.install_path is None:
            raise ValueError("install_path must be set for installation.")

        try:
            os.makedirs(self.install_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create installation directory at {self.install_path}: {e}") from e

        if not os.access(self.install_path, os.W_OK):
            raise PermissionError(f"The installation path {self.install_path} " "is not writable.")

        super().install(test_templates)
        self._write_config()

    def uninstall(self, test_templates: Iterable[TestTemplate]) -> None:
        """
        Uninstall the benchmarks or test templates from the installation path and remove the configuration file.

        This method does not require the installation path to be set in advance.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                be uninstalled.
        """
        super().uninstall(test_templates)

        config = self._read_config()
        install_path = config.get("install_path")
        if install_path:
            shutil.rmtree(install_path)
        self._remove_config()
