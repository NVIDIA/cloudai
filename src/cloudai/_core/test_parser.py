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
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from .base_multi_file_parser import BaseMultiFileParser
from .registry import Registry
from .test import Test, TestDefinition
from .test_template import TestTemplate


class TestParser(BaseMultiFileParser):
    """
    Parser for Test objects.

    Attributes
        test_template_mapping (Dict[str, TestTemplate]): Mapping of test template names to TestTemplate objects.
    """

    __test__ = False

    def __init__(
        self,
        directory_path: Path,
        test_template_mapping: Dict[str, TestTemplate],
    ) -> None:
        """
        Initialize the TestParser instance.

        Args:
            directory_path (str): Path to the directory containing test data.
            test_template_mapping (Dict[str, TestTemplate]): Mapping of test template names to TestTemplate objects.
        """
        super().__init__(directory_path)
        self.test_template_mapping: Dict[str, TestTemplate] = test_template_mapping

    @staticmethod
    def load_test_definition(data: dict) -> TestDefinition:
        test_template_name = data.get("test_template_name", "")
        registry = Registry()
        if test_template_name not in registry.test_definitions_map:
            raise NotImplementedError(f"TestTemplate with name '{test_template_name}' not supported.")

        try:
            test_def = registry.test_definitions_map[test_template_name].model_validate(data)
        except ValidationError as e:
            for err in e.errors(include_url=False):
                logging.error(
                    f"Field '{'.'.join(str(v) for v in err['loc'])}' with value"
                    f" '{err['input']}' is invalid: {err['msg']}"
                )
            raise ValueError("Failed to parse test spec") from e

        return test_def

    def _parse_data(self, data: Dict[str, Any]) -> Test:
        """
        Parse data for a Test object.

        Args:
            data (Dict[str, Any]): Data from a source (e.g., a TOML file).

        Returns:
            Test: Parsed Test object.
        """
        test_template_name = data.get("test_template_name", "")
        test_template = self.test_template_mapping.get(test_template_name)

        if not test_template:
            test_name = data.get("name", "Unnamed Test")
            raise ValueError(
                f"Test template with name '{test_template_name}' not found for test '{test_name}'. Please ensure the "
                f"test_template_name field in your test schema file matches one of the available test templates in "
                f"the provided test template directory. To resolve this issue, you can either add a corresponding "
                f"test template TOML file for '{test_template_name}' in the directory or remove the test schema file "
                f"that references this non-existing test template."
            )
        test_def = self.load_test_definition(data)

        env_vars = {}  # this field doesn't exist in Test or TestTemplate TOMLs
        """
        There are:
        1. global_env_vars, used in System
        2. extra_env_vars, used in Test
        """
        cmd_args = test_def.cmd_args.model_dump()
        extra_env_vars = test_def.extra_env_vars
        extra_cmd_args = test_def.extra_args_str()

        return Test(
            name=test_def.name,
            description=data.get("description", ""),
            test_template=test_template,
            env_vars=env_vars,
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args=extra_cmd_args,
        )

    def _parse_cmd_args(self, cmd_args_str: str) -> List[str]:
        """
        Parse a string of command-line arguments into a list.

        Args:
            cmd_args_str (str): Command-line arguments as a single string.

        Returns:
            List[str]: List of command-line arguments.
        """
        return cmd_args_str.split() if cmd_args_str else []
