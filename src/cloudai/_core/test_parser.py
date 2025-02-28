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
from pydantic import BaseModel, ValidationError

from .command_gen_strategy import CommandGenStrategy
from .exceptions import TestConfigParsingError, format_validation_error
from .grading_strategy import GradingStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .job_status_retrieval_strategy import JobStatusRetrievalStrategy
from .json_gen_strategy import JsonGenStrategy
from .registry import Registry
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

    @staticmethod
    def model_extras(m: BaseModel, prefix="cmd_args") -> set[str]:
        if m.model_extra is None:
            return set()

        extras = set()
        for field in m.model_fields:
            if isinstance(m.__dict__[field], BaseModel):
                extras |= TestParser.model_extras(m.__dict__[field], prefix=f"{prefix}.{field}")
        return extras | set([f"{prefix}.{k}" for k in m.model_extra])

    def load_test_definition(self, data: dict, strict: bool = False) -> TestDefinition:
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

        if strict and self.model_extras(test_def.cmd_args):
            logging.error(f"Strict check failed for test spec: '{self.current_file}'")
            for field in self.model_extras(test_def.cmd_args):
                logging.error(f"Unexpected field '{field}' in test spec.")
            raise TestConfigParsingError("Failed to parse test spec using strict mode")

        return test_def

    def _fetch_strategy(  # noqa: D417
        self,
        strategy_interface: Type[
            Union[
                TestTemplateStrategy,
                JobIdRetrievalStrategy,
                JobStatusRetrievalStrategy,
                GradingStrategy,
            ]
        ],
        system_type: Type[System],
        test_definition_type: Type[TestDefinition],
        cmd_args: Dict[str, Any],
    ) -> Optional[
        Union[
            TestTemplateStrategy,
            JobIdRetrievalStrategy,
            JobStatusRetrievalStrategy,
            GradingStrategy,
        ]
    ]:
        """
        Fetch a strategy from the registry based on system and template.

        Args:
            strategy_interface (Type[Union[TestTemplateStrategy,
                JobIdRetrievalStrategy, JobStatusRetrievalStrategy]]):
                The strategy interface to fetch.
            system_type (Type[System]): The system type.
            test_template_type (Type[TestTemplate]): The test template type.
            cmd_args (Dict[str, Any]): Command-line arguments.

        Returns:
            An instance of the requested strategy, or None.
        """
        key = (strategy_interface, system_type, test_definition_type)
        registry = Registry()
        strategy_type = registry.strategies_map.get(key)
        if strategy_type:
            if issubclass(strategy_type, TestTemplateStrategy):
                return strategy_type(self.system, cmd_args)
            else:
                return strategy_type()

        logging.debug(
            f"No {strategy_interface.__name__} found for "
            f"{test_definition_type.__name__} and "
            f"{type(self.system).__name__}"
        )
        return None

    def _get_test_template(self, name: str, tdef: TestDefinition) -> TestTemplate:
        """
        Dynamically retrieves the appropriate TestTemplate subclass based on the given name.

        Args:
            name (str): The name of the test template.
            tdef (TestDefinition): The test definition.

        Returns:
            Type[TestTemplate]: A subclass of TestTemplate corresponding to the given name.
        """
        cmd_args = tdef.cmd_args_dict

        obj = TestTemplate(system=self.system, name=name)
        obj.command_gen_strategy = cast(
            CommandGenStrategy,
            self._fetch_strategy(CommandGenStrategy, type(obj.system), type(tdef), cmd_args),
        )
        obj.json_gen_strategy = cast(
            JsonGenStrategy,
            self._fetch_strategy(JsonGenStrategy, type(obj.system), type(tdef), cmd_args),
        )
        obj.job_id_retrieval_strategy = cast(
            JobIdRetrievalStrategy,
            self._fetch_strategy(JobIdRetrievalStrategy, type(obj.system), type(tdef), cmd_args),
        )
        obj.job_status_retrieval_strategy = cast(
            JobStatusRetrievalStrategy,
            self._fetch_strategy(JobStatusRetrievalStrategy, type(obj.system), type(tdef), cmd_args),
        )
        obj.grading_strategy = cast(
            GradingStrategy, self._fetch_strategy(GradingStrategy, type(obj.system), type(tdef), cmd_args)
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

        test_template_name = data.get("test_template_name", "")
        test_template = self._get_test_template(test_template_name, test_def)

        return Test(test_definition=test_def, test_template=test_template)
