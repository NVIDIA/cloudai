# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, List, NamedTuple, Optional

import toml
from pydantic import ValidationError

from .core import Registry, System, TestConfigParsingError, format_validation_error
from .models.workload import TestDefinition
from .toml_utils import format_toml_decode_error


class TestConfigFailure(NamedTuple):  # noqa: D101
    declared_name: Optional[str]
    error: TestConfigParsingError


def failure_declaring(failures: Dict[Path, TestConfigFailure], test_name: str) -> Optional[TestConfigFailure]:
    return next((failure for failure in failures.values() if failure.declared_name == test_name), None)


def raise_on_any_failure(failures: Dict[Path, TestConfigFailure]) -> None:
    for failure in failures.values():
        raise failure.error


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

    def parse_all(self) -> tuple[List[Any], Dict[Path, TestConfigFailure]]:
        objects: List[Any] = []
        failures: Dict[Path, TestConfigFailure] = {}
        seen_names: Dict[str, Path] = {}
        for f in self.test_tomls:
            self.current_file = f
            logging.debug(f"Parsing file: {f}")
            declared_name: Optional[str] = None
            try:
                with f.open() as fh:
                    data: Dict[str, Any] = load_test_toml_file(fh, f)
                    declared_name = data.get("name")
                    parsed_object = self._parse_data(data)
            except TestConfigParsingError as e:
                failures[f] = TestConfigFailure(declared_name, e)
                continue
            obj_name: str = parsed_object.name
            if obj_name in seen_names:
                raise ValueError(f"Duplicate test name '{obj_name}' found in:\n  - {seen_names[obj_name]}\n  - {f}")
            seen_names[obj_name] = f
            objects.append(parsed_object)
        return objects, failures

    def load_test_definition(self, data: dict) -> TestDefinition:
        test_template_name = data.get("test_template_name")
        registry = Registry()
        if not test_template_name or test_template_name not in registry.test_definitions_map:
            raise TestConfigParsingError(
                f"Failed to parse test spec '{self.current_file}': "
                f"TestTemplate with name '{test_template_name}' not supported."
            )

        try:
            test_def = registry.test_definitions_map[test_template_name].model_validate(data)
        except ValidationError as e:
            reasons = "; ".join(format_validation_error(err) for err in e.errors(include_url=False))
            raise TestConfigParsingError(f"Failed to parse test spec '{self.current_file}': {reasons}") from e

        return test_def

    def _parse_data(self, data: Dict[str, Any]) -> TestDefinition:
        """
        Parse data for a Test object.

        Args:
            data (Dict[str, Any]): Data from a source (e.g., a TOML file).

        Returns:
            Test: Parsed Test object.
        """
        return self.load_test_definition(data)


def load_test_toml_file(fh, file_path: Path) -> Dict[str, Any]:
    try:
        return toml.load(fh)
    except toml.TomlDecodeError as e:
        raise TestConfigParsingError(format_toml_decode_error(file_path, e, "test spec")) from e
