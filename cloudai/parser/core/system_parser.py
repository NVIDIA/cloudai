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

import os
from typing import Callable, Dict, Type

import toml

from cloudai.schema.core import System

from .base_system_parser import BaseSystemParser


class SystemParser:
    """
    Parser for parsing system configurations.

    Attributes:
        _parsers (Dict[str, Type[BaseSystemParser]]): A mapping from system
            types to their corresponding parser classes.
        file_path (str): The file path to the system configuration file.
    """

    _parsers = {}

    def __init__(self, file_path: str):
        """
        Initializes a SystemParser instance.

        Args:
            file_path (str): The file path to the system configuration file.
        """
        self.file_path: str = file_path

    @classmethod
    def register(cls, system_type: str) -> Callable:
        """
        A decorator to register parser subclasses for specific system types.

        Args:
            system_type (str): The system type the parser can handle.

        Returns:
            Callable: A decorator function that registers the parser class.
        """

        def decorator(
            parser_class: Type[BaseSystemParser],
        ) -> Type[BaseSystemParser]:
            cls._parsers[system_type] = parser_class
            return parser_class

        return decorator

    @classmethod
    def get_supported_systems(cls) -> Dict[str, Type[BaseSystemParser]]:
        """
        Returns the supported system types and their corresponding parser classes.

        Returns:
            Dict[str, Type[BaseSystemParser]]: A dictionary of system types and
                                               their parser classes.
        """
        return cls._parsers

    def parse(self) -> System:
        """
        Parses the system configuration file, identifying the scheduler type
        and invoking the appropriate parser for further processing.

        Raises:
            FileNotFoundError: If the file path does not exist or is not a file.
            KeyError: If the 'scheduler' key is missing from the configuration.
            ValueError: If the 'scheduler' value is unsupported.

        Returns:
            System: The parsed system object.
        """
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

        with open(self.file_path, "r") as file:
            data = toml.load(file)
            scheduler = data.get("scheduler", "").lower()
            if scheduler not in self.get_supported_systems():
                raise ValueError(
                    f"Unsupported system type '{scheduler}'. "
                    f"Supported types: {', '.join(self.get_supported_systems())}"
                )
            parser_class = self._parsers.get(scheduler)
            if parser_class is None:
                raise NotImplementedError(f"No parser registered for system type: {scheduler}")
            parser = parser_class()
            return parser.parse(data)
