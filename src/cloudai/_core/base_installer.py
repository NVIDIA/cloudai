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

from .install_status_result import InstallStatusResult
from .system import System
from .test_template import TestTemplate


class BaseInstaller:
    """
    Base class for an Installer that manages the installation and uninstallation of benchmarks or test templates.

    This class provides a framework for checking if the necessary components are installed,
    installs them if necessary, and supports uninstallation.

    Attributes
        system (System): The system schema object.
    """

    def __init__(self, system: System):
        """
        Initialize the BaseInstaller with a system object.

        Args:
            system (System): The system schema object.
        """
        self.system = system
        logging.debug(f"BaseInstaller initialized for {self.system.scheduler}.")

    def _is_binary_installed(self, binary_name: str) -> bool:
        """
        Check if a given binary is installed on the system.

        Args:
            binary_name (str): The name of the binary to check.

        Returns:
            bool: True if the binary is installed, False otherwise.
        """
        logging.debug(f"Checking if binary '{binary_name}' is installed.")
        return shutil.which(binary_name) is not None

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check if common prerequisites are installed.

        This method should be overridden in derived classes for specific prerequisite checks.

        Returns
            InstallStatusResult: Result containing the status and any error message.
        """
        logging.debug("Checking for common prerequisites.")
        return InstallStatusResult(True)

    def is_installed(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Check if the necessary components for the provided test templates are already installed.

        Verify the installation status of each test template.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates to check for installation.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        not_installed = {}
        for test_template in test_templates:
            logging.debug(f"Verifying installation status of test template: {test_template.name}.")
            result = test_template.is_installed()
            if not result.success:
                not_installed[test_template.name] = result.message

        if not_installed:
            return InstallStatusResult(False, "Some test templates are not installed.", not_installed)
        else:
            return InstallStatusResult(True, "All test templates are installed.")

    def install(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Install the necessary components if they are not already installed.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        logging.debug("Starting installation of test templates.")
        prerequisites_result = self._check_prerequisites()
        if not prerequisites_result.success:
            return InstallStatusResult(False, "Prerequisites check failed.", {"error": prerequisites_result.message})

        install_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(template.install): template for template in test_templates}
            total, done = len(futures), 0
            for future in as_completed(futures):
                test_template = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        install_results[test_template.name] = "Success"
                    else:
                        install_results[test_template.name] = result.message
                    done += 1
                    logging.info(
                        f"{done}/{total} Installation for {test_template.name} finished with status: "
                        f"{result.message if result.message else 'OK'}"
                    )
                except Exception as e:
                    done += 1
                    logging.error(f"{done}/{total} Installation failed for {test_template.name}: {e}")
                    install_results[test_template.name] = str(e)

        all_success = all(result == "Success" for result in install_results.values())
        if all_success:
            return InstallStatusResult(True, "All test templates installed successfully.", install_results)
        else:
            return InstallStatusResult(False, "Some test templates failed to install.", install_results)

    def uninstall(self, test_templates: Iterable[TestTemplate]) -> InstallStatusResult:
        """
        Uninstall the benchmarks or test templates.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        logging.info("Uninstalling test templates.")
        uninstall_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(template.uninstall): template for template in test_templates}
            for future in as_completed(futures):
                test_template = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        uninstall_results[test_template.name] = "Success"
                    else:
                        uninstall_results[test_template.name] = result.message
                except Exception as e:
                    logging.error(f"Uninstallation failed for {test_template.name}: {e}")
                    uninstall_results[test_template.name] = str(e)

        all_success = all(result == "Success" for result in uninstall_results.values())
        if all_success:
            return InstallStatusResult(True, "All test templates uninstalled successfully.", uninstall_results)
        else:
            return InstallStatusResult(False, "Some test templates failed to uninstall.", uninstall_results)
