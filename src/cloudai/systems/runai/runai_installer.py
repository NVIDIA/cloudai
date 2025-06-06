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

from cloudai.core import BaseInstaller, Installable, InstallStatusResult

from .runai_system import RunAISystem


class RunAIInstaller(BaseInstaller):
    """Installer for RunAI systems."""

    def __init__(self, system: RunAISystem):
        super().__init__(system)

    def _check_prerequisites(self) -> InstallStatusResult:
        logging.info("Checking prerequisites for RunAI installation.")
        return InstallStatusResult(True)

    def install_one(self, item: Installable) -> InstallStatusResult:
        logging.info(f"Installing {item} for RunAI.")
        return InstallStatusResult(True)

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        logging.info(f"Uninstalling {item} for RunAI.")
        return InstallStatusResult(True)

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        logging.info(f"Checking if {item} is installed for RunAI.")
        return InstallStatusResult(True)

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        logging.info(f"Marking {item} as installed for RunAI.")
        return InstallStatusResult(True)
