# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, final

from cloudai.util import prepare_output_dir

from .install_status_result import InstallStatusResult
from .installables import Installable
from .system import System

TASK_LIMIT_THRESHOLD = 256


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
        self._low_thread_env = None
        logging.debug(f"BaseInstaller initialized for {self.system.scheduler}.")

    @property
    def is_low_thread_environment(self, threshold: int = TASK_LIMIT_THRESHOLD) -> bool:
        """
        Check if the current environment has a limit on the number of threads that is below the threshold.

        Args:
            threshold (int, optional): The threshold to consider "low thread". Defaults to TASK_LIMIT_THRESHOLD.

        Returns:
            bool: True if the environment has a low thread limit, False otherwise.
        """
        if self._low_thread_env is None:
            self._low_thread_env = self._check_low_thread_environment(threshold)
        return self._low_thread_env

    def _check_low_thread_environment(self, threshold: int = TASK_LIMIT_THRESHOLD) -> bool:
        try:
            result = subprocess.run(
                ["systemctl", "show", f"user-{os.getuid()}.slice", "--property=TasksMax"],
                capture_output=True,
                text=True,
                check=True,
            )
            _, value = result.stdout.strip().split("=", 1)
            value = value.strip()
            if value.lower() == "infinity":
                return False
            is_low_thread = int(value) < threshold
            if is_low_thread:
                logging.info("Low thread environment detected.")
            return is_low_thread
        except Exception as e:
            logging.debug(f"Could not determine TasksMax from systemd: {e}")
            return False

    @property
    def num_workers(self) -> Optional[int]:
        """
        Get the appropriate number of worker threads based on the environment.

        Returns:
            Optional[int]: 1 for low thread environments, None otherwise (allowing ThreadPoolExecutor to choose).
        """
        return 1 if self.is_low_thread_environment else None

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

    def all_items(self, items: Iterable[Installable], with_duplicates: bool = False) -> list[Installable]:
        all_items = list(items) + self.system.system_installables()
        return list(set(all_items)) if not with_duplicates else all_items

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
        elif not prepare_output_dir(self.system.hf_home_path):
            return InstallStatusResult(False, f"Error preparing hf home dir '{self.system.hf_home_path.absolute()}'")

        install_results: dict[Installable, InstallStatusResult] = {}
        for item in self.all_items(items):
            logging.debug(f"Installation check for {item!r}")
            result = self.is_installed_one(item)
            logging.debug(f"Installation check for {item!r}: {result.success}, {result.message}")
            install_results[item] = result

        self._populate_successful_install(items, install_results)

        nfailed = len([result for result in install_results.values() if not result.success])
        if nfailed:
            res = InstallStatusResult(
                False,
                f"{nfailed} item(s) are not installed.",
                {k: v for k, v in install_results.items() if not v.success},
            )
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
        elif not prepare_output_dir(self.system.hf_home_path):
            return InstallStatusResult(False, f"Error preparing hf home dir '{self.system.hf_home_path.absolute()}'")

        logging.debug(f"Going to install {len(set(items))} uniq item(s) (total is {len(list(items))})")

        install_results: dict[Installable, InstallStatusResult] = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.install_one, item): item for item in self.all_items(items)}
            total, done = len(futures), 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    done += 1
                    msg = f"{done}/{total} Installation of {item!r}: {result.message if result.message else 'OK'}"
                    install_results[item] = result
                    if result.success:
                        logging.info(msg)
                    else:
                        logging.error(msg)
                except Exception as e:
                    done += 1
                    logging.error(f"{done}/{total} Installation failed for {item!r}: {e}")
                    install_results[item] = InstallStatusResult(False, str(e))

        self._populate_successful_install(items, install_results)

        all_success = all(result.success for result in install_results.values())
        if all_success:
            return InstallStatusResult(True, "All items installed successfully.", install_results)

        nfailed = len([result for result in install_results.values() if not result.success])
        return InstallStatusResult(False, f"{nfailed} item(s) failed to install.", install_results)

    def _populate_successful_install(
        self, items: Iterable[Installable], install_results: dict[Installable, InstallStatusResult]
    ):
        for item in self.all_items(items, with_duplicates=True):
            if item not in install_results or not install_results[item].success:
                continue
            self.mark_as_installed_one(item)

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

        uninstall_results: dict[Installable, InstallStatusResult] = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.uninstall_one, item): item for item in self.all_items(items)}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    uninstall_results[item] = result
                except Exception as e:
                    logging.error(f"Uninstallation failed for {item!r}: {e}")
                    uninstall_results[item] = InstallStatusResult(False, str(e))

        all_success = all(result.success for result in uninstall_results.values())
        if all_success:
            return InstallStatusResult(True, "All items uninstalled successfully.", uninstall_results)

        nfailed = len([result for result in uninstall_results.values() if not result.success])
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
        install_results: dict[Installable, InstallStatusResult] = {}
        for item in self.all_items(items):
            result = self.mark_as_installed_one(item)
            install_results[item] = result

        self._populate_successful_install(items, install_results)

        return InstallStatusResult(True, "All items marked as installed successfully.", install_results)

    @abstractmethod
    def install_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def uninstall_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def is_installed_one(self, item: Installable) -> InstallStatusResult: ...

    @abstractmethod
    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult: ...
