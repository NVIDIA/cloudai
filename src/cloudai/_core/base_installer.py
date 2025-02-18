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

import logging
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, final

from cloudai.util import prepare_output_dir

from .install_status_result import InstallStatusResult
from .installables import Installable
from .system import System


class BaseInstaller(ABC):
    """
    Base class for an Installer that manages the installation and uninstallation of installable items.

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

    @final
    def is_installed(self, items: Iterable[Installable]) -> InstallStatusResult:
        """
        Check if the installable items are installed.

        Verify the installation status of each item.

        Args:
            items (Iterable[Installable]): Items to check for installation.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        if not prepare_output_dir(self.system.install_path):
            return InstallStatusResult(False, f"Error preparing install dir '{self.system.install_path.absolute()}'")

        not_installed = {}
        for item in items:
            logging.debug(f"Installation check for {item!r}")
            result = self.is_installed_one(item)
            logging.debug(f"Installation check for {item!r}: {result.success}, {result.message}")
            if not result.success:
                not_installed[item] = result.message

        if not_installed:
            res = InstallStatusResult(False, f"{len(not_installed)} item(s) are not installed.", not_installed)
            logging.debug(str(res))
            return res
        return InstallStatusResult(True, "All test templates are installed.")

    @final
    def install(self, items: Iterable[Installable]) -> InstallStatusResult:
        """
        Install the necessary components if they are not already installed.

        Args:
            items (Iterable[TestTemplate]): items to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        prerequisites_result = self._check_prerequisites()
        if not prerequisites_result.success:
            return prerequisites_result

        if not prepare_output_dir(self.system.install_path):
            return InstallStatusResult(False, f"Error preparing install dir '{self.system.install_path.absolute()}'")

        logging.debug(f"Going to install {len(set(items))} uniq item(s) (total is {len(list(items))})")
        logging.info(f"Going to install {len(set(items))} item(s)")

        install_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.install_one, item): item for item in set(items)}
            total, done = len(futures), 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    done += 1
                    msg = f"{done}/{total} Installation of {item!r}: {result.message if result.message else 'OK'}"
                    if result.success:
                        install_results[item] = "Success"
                        logging.info(msg)
                    else:
                        install_results[item] = result.message
                        logging.error(msg)
                except Exception as e:
                    done += 1
                    logging.error(f"{done}/{total} Installation failed for {item!r}: {e}")
                    install_results[item] = str(e)

        all_success = all(result == "Success" for result in install_results.values())
        if all_success:
            return InstallStatusResult(True, "All items installed successfully.", install_results)

        nfailed = len([result for result in install_results.values() if result != "Success"])
        return InstallStatusResult(False, f"{nfailed} item(s) failed to install.", install_results)

    @final
    def uninstall(self, items: Iterable[Installable]) -> InstallStatusResult:
        """
        Uninstall installable items.

        Args:
            items (Iterable[Installable]): Items to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        logging.debug(f"Going to uninstall {len(set(items))} uniq items (total {len(list(items))}).")
        logging.info(f"Going to uninstall {len(set(items))} items.")
        uninstall_results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.uninstall_one, item): item for item in set(items)}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        uninstall_results[item] = "Success"
                    else:
                        uninstall_results[item] = result.message
                except Exception as e:
                    logging.error(f"Uninstallation failed for {item!r}: {e}")
                    uninstall_results[item] = str(e)

        all_success = all(result == "Success" for result in uninstall_results.values())
        if all_success:
            return InstallStatusResult(True, "All items uninstalled successfully.", uninstall_results)

        nfailed = len([result for result in uninstall_results.values() if result != "Success"])
        return InstallStatusResult(False, f"{nfailed} item(s) failed to uninstall.", uninstall_results)

    @final
    def mark_as_installed(self, items: Iterable[Installable]) -> InstallStatusResult:
        """
        Mark the installable items as installed.

        Args:
            items (Iterable[Installable]): Items to mark as installed.

        Returns:
            InstallStatusResult: Result containing the status and error message if any.
        """
        install_results = {}
        for item in items:
            self.mark_as_installed_one(item)

        return InstallStatusResult(True, "All items marked as installed successfully.", install_results)

    @abstractmethod
    def install_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def uninstall_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def is_installed_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult: ...
