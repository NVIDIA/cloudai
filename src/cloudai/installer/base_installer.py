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

from cloudai.schema.core import System, TestTemplate


class BaseInstaller:
    """
    Base class for an Installer that manages the installation and
    uninstallation of benchmarks or test templates. This class provides
    a framework for checking if the necessary components are installed,
    installs them if necessary, and supports uninstallation.

    Attributes:
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

    def _check_prerequisites(self) -> None:
        """
        Check if common prerequisites are installed. This method should be
        overridden in derived classes for specific prerequisite checks.

        Raises:
            EnvironmentError: If a required binary is not installed.
        """
        self.logger.info("Checking for common prerequisites.")

    def is_installed(self, test_templates: Iterable[TestTemplate]) -> bool:
        """
        Check if the necessary components for the provided test templates
        are already installed by verifying the installation status of each
        test template.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                check for installation.

        Returns:
            bool: True if all test templates are installed, False otherwise.
        """
        self.logger.info("Verifying installation status of test templates.")
        return all(test_template.is_installed() for test_template in test_templates)

    def install(self, test_templates: Iterable[TestTemplate]) -> None:
        """
        Installs the necessary components if they are not already installed.
        Raises an exception if installation fails for any component.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates.
        """
        self.logger.info("Starting installation of test templates.")
        self._check_prerequisites()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(test_template.install) for test_template in test_templates]
            for future in as_completed(futures):
                future.result()

    def uninstall(self, test_templates: Iterable[TestTemplate]) -> None:
        """
        Uninstalls the benchmarks or test templates.
        Raises an exception if uninstallation fails for any component.

        Args:
            test_templates (Iterable[TestTemplate]): The test templates.
        """
        self.logger.info("Uninstalling test templates.")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(test_template.uninstall) for test_template in test_templates]
            for future in as_completed(futures):
                future.result()
