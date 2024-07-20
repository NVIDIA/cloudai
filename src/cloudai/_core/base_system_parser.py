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
from typing import Any, Dict

from .system import System


class BaseSystemParser(ABC):
    """
    Abstract base class for system parsers.

    Parses system configuration data and creates system objects.

    Methods
        parse: Abstract method to parse configuration data and return a System object.
    """

    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> System:
        """
        Parse configuration data and returns a System object.

        Args:
            data (Dict[str, Any]): The configuration data.

        Returns:
            System: A System object created from the configuration data.
        """
        pass
