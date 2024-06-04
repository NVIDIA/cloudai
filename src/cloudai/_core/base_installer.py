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

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from cloudai._core.install_status_result import InstallStatusResult

from .system import System
from .test_template import TestTemplate


class BaseInstaller:
    """
    Base class for an Installer that manages the installation and uninstallation of benchmarks or test templates.

    This class provides a framework for checking if the necessary components are installed,
    installs them if necessary, and supports uninstallation.

    Attributes
        system (System): The system schema object.
        logger (logging.Logger): Logger for capturing installation activities.
    """

    def __init__(self, system: System):
        """
        Initialize the BaseInstaller with a system object.

        Args:
            system (System): The system schema object.
        """
        self.system = system
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"BaseInstaller initialized for {self.system.scheduler}.")

    def _is_binary_installed(self, binary_name: str) -> bool:
        """
        Check if a given binary is installed on the system.

        Args:
            binary_name (str): The name of the binary to check.

        Returns:
            bool: True if the binary is installed, False otherwise.
        """
        self.logger.debug(f"Checking if binary '{binary_name}' is installed.")
        return shutil.which(binary_name) is not None

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check if common prerequisites are installed.

        This method should be overridden in derived classes for specific prerequisite checks.

        Returns
            InstallStatusResult: Result containing the status and any error message.
        """
        self.logger.info("Checking for common prerequisites.")
        return InstallStatusResult(True)

    def is_installed(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Check if the necessary components for the provided test templates are already installed.

        Verify the installation status of each test template.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to check for installation.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        self.logger.info("Verifying installation status of test templates.")
        not_installed = {}
        for test_template in test_templates:
            try:
                if not test_template.is_installed():
                    not_installed[test_template.name] = "Not installed"
            except Exception as e:
                not_installed[test_template.name] = str(e)

        if not_installed:
            return InstallStatusResult(False, "Some test templates are not installed.", not_installed)
        else:
            return InstallStatusResult(True, "All test templates are installed.")

    def install(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Install the necessary components if they are not already installed.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        self.logger.info("Starting installation of test templates.")
        prerequisites_result = self._check_prerequisites()
        if not prerequisites_result:
            return prerequisites_result

        install_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(test_template.install): test_template for test_template in test_templates}
            for future in as_completed(futures):
                test_template = futures[future]
                try:
                    future.result()
                    install_results[test_template.name] = "Success"
                except Exception as e:
                    self.logger.error(f"Installation failed for {test_template.name}: {e}")
                    install_results[test_template.name] = str(e)

        all_success = all(result == "Success" for result in install_results.values())
        if all_success:
            return InstallStatusResult(True, "All test templates installed successfully.")
        else:
            return InstallStatusResult(False, "Some test templates failed to install.", install_results)

    def uninstall(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Uninstall the benchmarks or test templates.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        self.logger.info("Uninstalling test templates.")
        uninstall_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(test_template.uninstall): test_template for test_template in test_templates}
            for future in as_completed(futures):
                test_template = futures[future]
                try:
                    future.result()
                    uninstall_results[test_template.name] = "Success"
                except Exception as e:
                    self.logger.error(f"Uninstallation failed for {test_template.name}: {e}")
                    uninstall_results[test_template.name] = str(e)

        all_success = all(result == "Success" for result in uninstall_results.values())
        if all_success:
            return InstallStatusResult(True, "All test templates uninstalled successfully.")
        else:
            return InstallStatusResult(False, "Some test templates failed to uninstall.", uninstall_results)
