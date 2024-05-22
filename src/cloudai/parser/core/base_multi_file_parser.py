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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import toml
from cloudai.schema.core.test import Test
from cloudai.schema.core.test_template import TestTemplate


class BaseMultiFileParser(ABC):
    """
    Abstract base class for test configuration parsers.

    Parses files in a given directory and creates objects based on the file contents.

    Attributes
        directory_path (str): Path to the directory with configuration files.
    """

    def __init__(self, directory_path: str):
        self.directory_path: str = directory_path

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
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".toml"):
                file_path: str = os.path.join(self.directory_path, filename)
                with open(file_path, "r") as file:
                    data: Dict[str, Any] = toml.load(file)
                    parsed_object = self._parse_data(data)
                    obj_name: str = parsed_object.name
                    if obj_name in objects:
                        raise ValueError(f"Duplicate name found: {obj_name}")
                    objects.append(parsed_object)
        return objects
