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


from typing import Dict

from .system import System


class BaseInstallerFrontend:
    """
    Base class for installer frontends that handle user interaction for installation.

    Attributes
        system (System): The system schema object.
    """

    def __init__(self, system: System) -> None:
        """
        Initialize the BaseInstallerFrontend with a system object.

        Args:
            system (System): The system schema object.
        """
        self.system: System = system

    def get_installation_args(self) -> Dict[str, str]:
        """
        Get installation arguments from the user.

        Returns
            Dict[str, str]: Dictionary of installation arguments.
        """
        raise NotImplementedError("Subclasses should implement this method.")
