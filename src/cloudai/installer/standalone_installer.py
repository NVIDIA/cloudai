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

from cloudai import BaseInstaller, InstallStatusResult


class StandaloneInstaller(BaseInstaller):
    """
    Installer for systems that do not use a scheduler (standalone systems).

    Handles the installation of benchmarks or test templates for standalone systems.
    """

    PREREQUISITES = ["ps", "kill"]

    def _check_prerequisites(self) -> InstallStatusResult:
        """Check for the presence of required binaries, returning an error status if any are missing."""
        super()._check_prerequisites()  # TODO: if fails, print out missing prerequisites
        missing_binaries = []
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                missing_binaries.append(binary)

        if missing_binaries:
            missing_str = ", ".join(missing_binaries)
            return InstallStatusResult(False, f"Required binaries not installed: {missing_str}.")

        return InstallStatusResult(True)
