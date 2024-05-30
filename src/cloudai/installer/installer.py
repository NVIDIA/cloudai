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
from typing import Iterable

from cloudai._core.base_installer import BaseInstaller
from cloudai._core.registry import Registry
from cloudai._core.system import System
from cloudai._core.test_template import TestTemplate


class Installer:
    """
    A wrapper class that creates and manages a specific installer instance based on the system's configuration.

    This class facilitates the initialization of the appropriate installer
    based on the system.

    Attributes
        _installers (dict): A class-level dictionary mapping scheduler
            types to their corresponding installer subclasses. Used for
            dynamic installer creation based on system configuration.
        installer (BaseInstaller): The specific installer instance for the system.
        logger (logging.Logger): Logger for capturing installation activities.
    """

    def __init__(self, system: System):
        """
        Initialize the Installer with a system object and installation path.

        Args:
            system (System): The system schema object.
        """
        self.logger = logging.getLogger(__name__ + ".Installer")
        self.logger.info("Initializing Installer with system configuration.")
        self.installer = self.create_installer(system)

    @classmethod
    def create_installer(cls, system: System) -> BaseInstaller:
        """
        Create an installer instance based on the system's scheduler type.

        Args:
            system (System): The system schema object.

        Returns:
            BaseInstaller: An instance of a subclass of BaseInstaller
                           suitable for the given system's scheduler.

        Raises:
            NotImplementedError: If no installer is available for the
                                 system's scheduler.
        """
        scheduler_type = system.scheduler
        registry = Registry()
        installer_class = registry.installers_map.get(scheduler_type)
        if installer_class is None:
            raise NotImplementedError(f"No installer available for scheduler: {scheduler_type}")
        return installer_class(system)

    def is_installed(self, test_templates: Iterable[TestTemplate]) -> bool:
        """
        Check if the necessary components for the provided test templates are already installed.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                check for installation.

        Returns:
            bool: True if all necessary components are installed, False otherwise.
        """
        self.logger.info("Checking installation status of components.")
        return self.installer.is_installed(test_templates)

    def install(self, test_templates: Iterable[TestTemplate]):
        """
        Check if the necessary components are installed and install them if not using the instantiated installer.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                install.
        """
        self.logger.info("Installing test templates.")
        self.installer.install(test_templates)

    def uninstall(self, test_templates: Iterable[TestTemplate]):
        """
        Uninstall the benchmarks or test templates using the instantiated installer.

        Args:
            test_templates (Iterable[TestTemplate]): The list of test templates to
                uninstall.
        """
        self.logger.info("Uninstalling test templates.")
        self.installer.uninstall(test_templates)
