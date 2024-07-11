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

import logging
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError

from .base_multi_file_parser import BaseMultiFileParser
from .test import Test
from .test_definitions import (
    ChakraReplayTestDefinition,
    NCCLTestDefinition,
    NeMoLauncherTestDefinition,
    SleepTestDefinition,
    UCCTestDefinition,
)
from .test_template import TestTemplate

TEST_DEFINITIONS = {
    "UCCTest": UCCTestDefinition,
    "NcclTest": NCCLTestDefinition,
    "ChakraReplay": ChakraReplayTestDefinition,
    "Sleep": SleepTestDefinition,
    "NeMoLauncher": NeMoLauncherTestDefinition,
}


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
        logging.debug(f"Content: {data}")

        if not test_template:
            raise ValueError(f"TestTemplate with name '{test_template_name}' not found.")

        if test_template_name not in TEST_DEFINITIONS:
            raise NotImplementedError(f"TestTemplate with name '{test_template_name}' not supported.")
        try:
            test_def = TEST_DEFINITIONS[test_template_name](**data)
        except ValidationError as e:
            for err in e.errors():
                logging.error(pprint.saferepr(err))
            raise ValueError(f"Failed to parse test definition") from e

        env_vars = {}  # data.get("env_vars", {})     # this field doesn't exist in Test or TestTemplate TOMLs
        """
        There are:
        1. global_env_vars, used in System
        2. extra_env_vars, used in Test
        """
        cmd_args = test_def.cmd_args.dict()
        extra_env_vars = test_def.extra_env_vars
        extra_cmd_args = test_def.extra_cmd_args

        # flattened_template_cmd_args = self._flatten_template_dict_keys(test_template.cmd_args)
        # self._validate_args(cmd_args, flattened_template_cmd_args)

        # flattened_template_env_vars = self._flatten_template_dict_keys(test_template.env_vars)
        # self._validate_args(env_vars, flattened_template_env_vars)

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

    def _flatten_template_dict_keys(self, nested_args: Dict[str, Any], parent_key: str = "") -> Set[str]:
        """
        Recursively flattens the nested dictionary structure from the test template.

        Includes keys with 'default' and 'values' as valid keys, while ignoring keys that specifically end with
        'default' or 'values'.

        Args:
            nested_args (Dict[str, Any]): Nested argument structure from the test template.
            parent_key (str): Parent key for nested arguments.

        Returns:
            Set[str]: Set of all valid argument keys.
        """
        keys = set()
        for k, v in nested_args.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if k in ["type", "values", "default"]:
                continue

            if isinstance(v, dict):
                if "default" in v:
                    keys.add(new_key)
                keys.update(self._flatten_template_dict_keys(v, new_key))
            else:
                keys.add(new_key)

        return keys

    def _validate_args(self, args: Dict[str, Any], valid_keys: Set[str]) -> None:
        """
        Validate the provided arguments against a set of valid keys.

        Args:
            args (Dict[str, Any]): Arguments provided in the TOML configuration.
            valid_keys (Set[str]): Set of valid keys from the flattened template arguments.

        Raises:
            ValueError: If an argument is not defined in the TestTemplate's arguments.
        """
        for arg_key in args:
            if arg_key not in valid_keys:
                raise ValueError(f"Argument '{arg_key}' is not defined in the TestTemplate's arguments.")
