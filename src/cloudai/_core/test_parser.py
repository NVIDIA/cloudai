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
from typing import Any, Dict, List, Optional, Type, Union, cast

import toml
from pydantic import ValidationError

from .command_gen_strategy import CommandGenStrategy
from .exceptions import TestConfigParsingError, format_validation_error
from .grading_strategy import GradingStrategy
from .install_strategy import InstallStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .job_status_retrieval_strategy import JobStatusRetrievalStrategy
from .json_gen_strategy import JsonGenStrategy
from .registry import Registry
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test import Test, TestDefinition
from .test_template import TestTemplate
from .test_template_strategy import TestTemplateStrategy


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
        test_template_name = data.get("test_template_name", "")
        registry = Registry()
        if test_template_name not in registry.test_definitions_map:
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

    def _fetch_strategy(  # noqa: D417
        self,
        strategy_interface: Type[
            Union[
                TestTemplateStrategy,
                ReportGenerationStrategy,
                JobIdRetrievalStrategy,
                JobStatusRetrievalStrategy,
                GradingStrategy,
            ]
        ],
        system_type: Type[System],
        test_template_type: Type[TestTemplate],
        cmd_args: Dict[str, Any],
    ) -> Optional[
        Union[
            TestTemplateStrategy,
            ReportGenerationStrategy,
            JobIdRetrievalStrategy,
            JobStatusRetrievalStrategy,
            GradingStrategy,
        ]
    ]:
        """
        Fetch a strategy from the registry based on system and template.

        Args:
            strategy_interface (Type[Union[TestTemplateStrategy, ReportGenerationStrategy,
                JobIdRetrievalStrategy, JobStatusRetrievalStrategy]]):
                The strategy interface to fetch.
            system_type (Type[System]): The system type.
            test_template_type (Type[TestTemplate]): The test template type.
            cmd_args (Dict[str, Any]): Command-line arguments.

        Returns:
            An instance of the requested strategy, or None.
        """
        key = (strategy_interface, system_type, test_template_type)
        registry = Registry()
        strategy_type = registry.strategies_map.get(key)
        if strategy_type:
            if issubclass(strategy_type, TestTemplateStrategy):
                return strategy_type(self.system, cmd_args)
            else:
                return strategy_type()

        logging.debug(
            f"No {strategy_interface.__name__} found for "
            f"{test_template_type.__name__} and "
            f"{type(self.system).__name__}"
        )
        return None

    def _get_test_template(self, name: str, cmd_args: Dict[str, Any]) -> TestTemplate:
        """
        Dynamically retrieves the appropriate TestTemplate subclass based on the given name.

        Args:
            name (str): The name of the test template.
            cmd_args (Dict[str, Any]): Command-line arguments.

        Returns:
            Type[TestTemplate]: A subclass of TestTemplate corresponding to the given name.
        """
        template_classes = Registry().test_templates_map

        test_template_class = template_classes.get(name)
        if not test_template_class:
            raise ValueError(f"Unsupported test_template name: {name}")

        obj = test_template_class(system=self.system, name=name)
        obj.install_strategy = cast(
            InstallStrategy, self._fetch_strategy(InstallStrategy, type(obj.system), type(obj), cmd_args)
        )
        obj.command_gen_strategy = cast(
            CommandGenStrategy,
            self._fetch_strategy(CommandGenStrategy, type(obj.system), type(obj), cmd_args),
        )
        obj.json_gen_strategy = cast(
            JsonGenStrategy,
            self._fetch_strategy(JsonGenStrategy, type(obj.system), type(obj), cmd_args),
        )
        obj.job_id_retrieval_strategy = cast(
            JobIdRetrievalStrategy,
            self._fetch_strategy(JobIdRetrievalStrategy, type(obj.system), type(obj), cmd_args),
        )
        obj.job_status_retrieval_strategy = cast(
            JobStatusRetrievalStrategy,
            self._fetch_strategy(JobStatusRetrievalStrategy, type(obj.system), type(obj), cmd_args),
        )
        obj.report_generation_strategy = cast(
            ReportGenerationStrategy,
            self._fetch_strategy(ReportGenerationStrategy, type(obj.system), type(obj), cmd_args),
        )
        obj.grading_strategy = cast(
            GradingStrategy, self._fetch_strategy(GradingStrategy, type(obj.system), type(obj), cmd_args)
        )
        return obj

    def _parse_data(self, data: Dict[str, Any]) -> Test:
        """
        Parse data for a Test object.

        Args:
            data (Dict[str, Any]): Data from a source (e.g., a TOML file).

        Returns:
            Test: Parsed Test object.
        """
        test_def = self.load_test_definition(data)

        """
        There are:
        1. global_env_vars, used in System
        2. extra_env_vars, used in Test
        """
        cmd_args = test_def.cmd_args_dict

        test_template_name = data.get("test_template_name", "")
        test_template = self._get_test_template(test_template_name, cmd_args)

        if not test_template:
            test_name = data.get("name", "Unnamed Test")
            raise ValueError(
                f"Test template with name '{test_template_name}' not found for test '{test_name}'. Please ensure the "
                f"test_template_name field in your test schema file matches one of the available test templates in "
                f"the provided test template directory. To resolve this issue, you can either add a corresponding "
                f"test template TOML file for '{test_template_name}' in the directory or remove the test schema file "
                f"that references this non-existing test template."
            )

        return Test(test_definition=test_def, test_template=test_template)

    def _parse_cmd_args(self, cmd_args_str: str) -> List[str]:
        """
        Parse a string of command-line arguments into a list.

        Args:
            cmd_args_str (str): Command-line arguments as a single string.

        Returns:
            List[str]: List of command-line arguments.
        """
        return cmd_args_str.split() if cmd_args_str else []
