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

from abc import abstractmethod

from .install_status_result import InstallStatusResult
from .test_template_strategy import TestTemplateStrategy


class InstallStrategy(TestTemplateStrategy):
    """Abstract base class defining the interface for installation strategies across different system environments."""

    @abstractmethod
    def is_installed(self) -> InstallStatusResult:
        """
        Check if the necessary components are already installed on the system.

        Returns
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        return InstallStatusResult(success=True)

    @abstractmethod
    def install(self) -> InstallStatusResult:
        """
        Perform installation operations for a specific system.

        Returns
            InstallStatusResult: Result containing the installation status and error message if installation failed.
        """
        return InstallStatusResult(success=True)

    @abstractmethod
    def uninstall(self) -> InstallStatusResult:
        """
        Perform uninstallation operations for a specific system.

        Returns
            InstallStatusResult: Result containing the uninstallation status and error message if uninstallation failed.
        """
        return InstallStatusResult(success=True)
