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

import os

import toml

from .registry import Registry
from .system import System


class SystemParser:
    """
    Parser for parsing system configurations.

    Attributes
        _parsers (Dict[str, Type[BaseSystemParser]]): A mapping from system types to their corresponding parser
            classes.
        file_path (str): The file path to the system configuration file.
    """

    _parsers = {}

    def __init__(self, file_path: str):
        """
        Initialize a SystemParser instance.

        Args:
            file_path (str): The file path to the system configuration file.
        """
        self.file_path: str = file_path

    def parse(self) -> System:
        """
        Parse the system configuration file, identifying the scheduler type and invoking the appropriate parser.

        Raises
            FileNotFoundError: If the file path does not exist or is not a file.
            KeyError: If the 'scheduler' key is missing from the configuration.
            ValueError: If the 'scheduler' value is unsupported.

        Returns
            System: The parsed system object.
        """
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

        with open(self.file_path, "r") as file:
            data = toml.load(file)
            scheduler = data.get("scheduler", "").lower()
            registry = Registry()
            if scheduler not in registry.system_parsers_map:
                raise ValueError(
                    f"Unsupported system type '{scheduler}'. "
                    f"Supported types: {', '.join(registry.system_parsers_map.keys())}"
                )
            parser_class = registry.system_parsers_map[scheduler]
            if parser_class is None:
                raise NotImplementedError(f"No parser registered for system type: {scheduler}")
            parser = parser_class()
            return parser.parse(data)
