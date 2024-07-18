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

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import toml

from .test import Test
from .test_template import TestTemplate


class BaseMultiFileParser(ABC):
    """
    Abstract base class for test configuration parsers.

    Parses files in a given directory and creates objects based on the file contents.

    Attributes
        directory_path (str): Path to the directory with configuration files.
    """

    def __init__(self, directory_path: Path):
        self.directory_path = directory_path

    @abstractmethod
    def _parse_data(self, data: Dict[str, Any]) -> Union[Test, TestTemplate]:
        """
        Abstract method to parse data from a TOML file.

        Must be implemented by subclasses to create specific types of objects.

        Args:
            data (Dict[str, Any]): Data parsed from a TOML file.

        Returns:
            Any: Object created from the parsed data.
        """
        pass

    def parse_all(self) -> List[Any]:
        """
        Parse all TOML files in the directory and returns a list of objects.

        Returns
            List[Any]: List of objects from the configuration files.
        """
        objects: List[Any] = []
        for f in self.directory_path.glob("*.toml"):
            logging.debug(f"Parsing file: {f}")
            with f.open() as fh:
                data: Dict[str, Any] = toml.load(fh)
                parsed_object = self._parse_data(data)
                obj_name: str = parsed_object.name
                if obj_name in objects:
                    raise ValueError(f"Duplicate name found: {obj_name}")
                objects.append(parsed_object)
        return objects
