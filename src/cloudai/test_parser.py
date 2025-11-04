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
from pathlib import Path
from typing import Any, Dict, List

import toml
from pydantic import ValidationError

from .core import Registry, System, Test, TestConfigParsingError, format_validation_error
from .models.workload import TestDefinition


class TestParser:
    """
    Parser for Test objects.

    Attributes
        test_template_mapping (Dict[str, TestTemplate]): Mapping of test template names to TestTemplate objects.
    """

    __test__ = False

    def __init__(self, test_tomls: list[Path], system: System) -> None:
        """
        Initialize the TestParser instance.

        Args:
            test_tomls (list[Path]): List of paths to test TOML files.
            system (System): The system object.
        """
        self.system = system
        self.test_tomls = test_tomls

    def parse_all(self) -> List[Any]:
        """
        Parse all TOML files in the directory and returns a list of objects.

        Returns
            List[Any]: List of objects from the configuration files.
        """
        objects: List[Any] = []
        for f in self.test_tomls:
            self.current_file = f
            logging.debug(f"Parsing file: {f}")
            with f.open() as fh:
                data: Dict[str, Any] = toml.load(fh)
                parsed_object = self._parse_data(data)
                obj_name: str = parsed_object.name
                if obj_name in objects:
                    raise ValueError(f"Duplicate name found: {obj_name}")
                objects.append(parsed_object)
        return objects

    def load_test_definition(self, data: dict) -> TestDefinition:
        test_template_name = data.get("test_template_name")
        registry = Registry()
        if not test_template_name or test_template_name not in registry.test_definitions_map:
            logging.error(f"Failed to parse test spec: '{self.current_file}'")
            logging.error(f"TestTemplate with name '{test_template_name}' not supported.")
            raise NotImplementedError(f"TestTemplate with name '{test_template_name}' not supported.")

        try:
            test_def = registry.test_definitions_map[test_template_name].model_validate(data)
        except ValidationError as e:
            logging.error(f"Failed to parse test spec: '{self.current_file}'")
            for err in e.errors(include_url=False):
                err_msg = format_validation_error(err)
                logging.error(err_msg)
            raise TestConfigParsingError("Failed to parse test spec") from e

        return test_def

    def _parse_data(self, data: Dict[str, Any]) -> Test:
        """
        Parse data for a Test object.

        Args:
            data (Dict[str, Any]): Data from a source (e.g., a TOML file).

        Returns:
            Test: Parsed Test object.
        """
        test_def = self.load_test_definition(data)
        return Test(test_definition=test_def)
