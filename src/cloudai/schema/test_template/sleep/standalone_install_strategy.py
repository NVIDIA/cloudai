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

import shutil

from cloudai import InstallStrategy


class SleepStandaloneInstallStrategy(InstallStrategy):
    """
    Installation strategy for the Sleep test on standalone systems.

    This strategy checks for the availability of the sleep command,
    which is typically pre-installed on UNIX-like systems.
    """

    def is_installed(self) -> bool:
        """
        Check if the sleep command is available on the system.

        Returns
            bool: True if the sleep command is found in the system's PATH,
                  False otherwise.
        """
        return shutil.which("sleep") is not None

    def install(self) -> None:
        """
        Verify if the sleep command is available in the system.

        Since sleep is a common command, this method mainly serves as a check
        rather than performing an actual installation.

        Raises
            RuntimeError: If the sleep command is not found in the system.
        """
        if shutil.which("sleep") is None:
            raise RuntimeError("Sleep command is not available on this system.")

    def uninstall(self) -> None:
        """
        Uninstall Sleep test.

        As the sleep command is a common system utility, this method does not perform any operations.
        """
        pass
