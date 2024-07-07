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

from typing import Dict, Optional


class InstallStatusResult:
    """
    Class representing the result of an installation, uninstallation, or status check.

    Attributes
        success (bool): Indicates whether the operation was successful.
        message (str): A message providing additional information about the result.
        details (Optional[Dict[str, str]]): A dictionary containing details about the result for each test template.
    """

    def __init__(self, success: bool, message: str = "", details: Optional[Dict[str, str]] = None):
        """
        Initialize the InstallStatusResult.

        Args:
            success (bool): Indicates whether the operation was successful.
            message (str): A message providing additional information about the result.
            details (Optional[Dict[str, str]]): A dictionary containing details about the result for each test template.
        """
        self.success = success
        self.message = message
        self.details = details if details else {}

    def __bool__(self):
        return self.success

    def __str__(self):
        details_str = "\n".join(f"  - {key}: {value}" for key, value in self.details.items())
        return f"{self.message}\n{details_str}" if self.details else self.message
