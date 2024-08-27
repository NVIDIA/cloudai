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
from typing import Optional


class ReportGenerationStrategy:
    """
    Abstract class for generating reports from directories.

    This class defines a strategy for checking if a strategy can handle a
    specific directory and for generating a report from that directory.
    """

    @abstractmethod
    def can_handle_directory(self, directory_path: str) -> bool:
        """
        Determine if the strategy can handle the directory.

        Args:
            directory_path (str): Path to the directory.

        Returns:
            bool: True if can handle, False otherwise.
        """
        pass

    @abstractmethod
    def generate_report(self, test_name: str, directory_path: str, sol: Optional[float] = None) -> None:
        """
        Generate a report from the directory.

        Args:
            test_name (str): The name of the test.
            directory_path (str): Path to the directory.
            sol (Optional[float]): Speed-of-light performance for reference.
        """
        pass
