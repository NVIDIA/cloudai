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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class NcclTestOutputReaderMixin(ABC):
    """Base mixin to provide a standardized interface for reading NCCL test output."""

    @abstractmethod
    def _get_stdout_content(self, directory_path: Path) -> Optional[str]:
        """
        Abstract method to retrieve the content of stdout based on the system.

        Args:
            directory_path (Path): Path to the directory containing the output files.

        Returns:
            Optional[str]: Content of the stdout file, or None if not found.
        """
        pass
