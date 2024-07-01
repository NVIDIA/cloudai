#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai import InstallStatusResult, InstallStrategy


class SleepStandaloneInstallStrategy(InstallStrategy):
    """
    Installation strategy for the Sleep test on standalone systems.

    This strategy checks for the availability of the sleep command,
    which is typically pre-installed on UNIX-like systems.
    """

    def is_installed(self) -> InstallStatusResult:
        """
        Check if the sleep command is available on the system.

        Returns
            InstallStatusResult: Status result indicating if the sleep command is found.
        """
        if shutil.which("sleep") is not None:
            return InstallStatusResult(success=True)
        return InstallStatusResult(success=False, message="Sleep command is not available on this system.")

    def install(self) -> InstallStatusResult:
        """
        Verify if the sleep command is available in the system.

        Since sleep is a common command, this method mainly serves as a check rather than performing an actual
        installation.

        Returns
            InstallStatusResult: Status result indicating success or failure of the check.

        Raises
            RuntimeError: If the sleep command is not found in the system.
        """
        if shutil.which("sleep") is None:
            return InstallStatusResult(success=False, message="Sleep command is not available on this system.")
        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        """
        Uninstall Sleep test.

        As the sleep command is a common system utility, this method does not perform any operations.

        Returns
            InstallStatusResult: Status result indicating the command is always successful.
        """
        return InstallStatusResult(success=True)
