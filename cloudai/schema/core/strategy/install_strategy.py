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

from abc import abstractmethod

from .test_template_strategy import TestTemplateStrategy


class InstallStrategy(TestTemplateStrategy):
    """
    Abstract base class defining the interface for installation strategies
    across different system environments. This class provides methods to check
    if necessary components are installed, to install those components, and
    to uninstall them if needed.
    """

    @abstractmethod
    def is_installed(self) -> bool:
        """
        Checks if the necessary components are already installed on the system.

        Returns:
            bool: True if the necessary components are installed, False otherwise.
        """
        pass

    @abstractmethod
    def install(self) -> None:
        """
        Performs installation operations for a specific system.

        Returns:
            None
        """
        pass

    @abstractmethod
    def uninstall(self) -> None:
        """
        Performs uninstallation operations for a specific system.

        Returns:
            None
        """
        pass
