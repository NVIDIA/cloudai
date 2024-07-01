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
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, cast

from .base_multi_file_parser import BaseMultiFileParser
from .command_gen_strategy import CommandGenStrategy
from .grading_strategy import GradingStrategy
from .install_strategy import InstallStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .job_status_retrieval_strategy import JobStatusRetrievalStrategy
from .registry import Registry
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test_template import TestTemplate
from .test_template_strategy import TestTemplateStrategy


class TestTemplateParser(BaseMultiFileParser):
    """
    Parser for creating TestTemplate objects from provided data.

    Attributes
        system (System): The system schema object.
    """

    __test__ = False

    VALID_DATA_TYPES = ["preset", "bool", "int", "str"]

    def __init__(self, system: System, directory_path: Path) -> None:
        """
        Initialize a TestTemplateParser with a specific system and directory path.

        Args:
            system (System): The system schema object.
            directory_path (str): The directory path where test templates are located.
        """
        super().__init__(directory_path)
        self.system = system
        self.directory_path = directory_path

    def _fetch_strategy(  # noqa: D417
        self,
        strategy_interface: Type[
            Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy, JobStatusRetrievalStrategy]
        ],
        system_type: Type[System],
        test_template_type: Type[TestTemplate],
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> Optional[
        Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy, JobStatusRetrievalStrategy]
    ]:
        """
        Fetch a strategy from the registry based on system and template.

        Args:
            strategy_interface (Type[Union[TestTemplateStrategy, ReportGenerationStrategy,
                JobIdRetrievalStrategy, JobStatusRetrievalStrategy]]):
                The strategy interface to fetch.
            system_type (Type[System]): The system type.
            test_template_type (Type[TestTemplate]): The test template type.
            env_vars (Dict[str, Any]): Environment variables.
            cmd_args (Dict[str, Any]): Command-line arguments.

        Returns:
            An instance of the requested strategy, or None.
        """
        key = (strategy_interface, system_type, test_template_type)
        registry = Registry()
        strategy_type = registry.strategies_map.get(key)
        if strategy_type:
            if issubclass(strategy_type, TestTemplateStrategy):
                return strategy_type(self.system, env_vars, cmd_args)
            else:
                return strategy_type()

        logging.warning(
            f"No {strategy_interface.__name__} found for " f"{type(self).__name__} and " f"{type(self.system).__name__}"
        )
        return None

    def _parse_data(self, data: Dict[str, Any]) -> TestTemplate:
        """
        Parse data for a TestTemplate object.

        Args:
            data (Dict[str, Any]): Data from a source (e.g., a TOML file).

        Returns:
            TestTemplate: Parsed TestTemplate object.
        """
        if "name" not in data:
            raise KeyError("The 'name' field is missing from the data.")
        name = data["name"]
        test_template_class = self._get_test_template_class(name)

        assert issubclass(test_template_class, TestTemplate), "Invalid test template class"

        env_vars = self._extract_args(data.get("env_vars", {}))
        cmd_args = self._extract_args(data.get("cmd_args", {}))

        self._validate_args(env_vars, "Environment")
        self._validate_args(cmd_args, "Command-line")

        obj = test_template_class(system=self.system, name=name, env_vars=env_vars, cmd_args=cmd_args)
        obj.install_strategy = cast(
            InstallStrategy, self._fetch_strategy(InstallStrategy, type(obj.system), type(obj), env_vars, cmd_args)
        )
        obj.command_gen_strategy = cast(
            CommandGenStrategy,
            self._fetch_strategy(CommandGenStrategy, type(obj.system), type(obj), env_vars, cmd_args),
        )
        obj.job_id_retrieval_strategy = cast(
            JobIdRetrievalStrategy,
            self._fetch_strategy(JobIdRetrievalStrategy, type(obj.system), type(obj), env_vars, cmd_args),
        )
        obj.job_status_retrieval_strategy = cast(
            JobStatusRetrievalStrategy,
            self._fetch_strategy(JobStatusRetrievalStrategy, type(obj.system), type(obj), env_vars, cmd_args),
        )
        obj.report_generation_strategy = cast(
            ReportGenerationStrategy,
            self._fetch_strategy(ReportGenerationStrategy, type(obj.system), type(obj), env_vars, cmd_args),
        )
        obj.grading_strategy = cast(
            GradingStrategy, self._fetch_strategy(GradingStrategy, type(obj.system), type(obj), env_vars, cmd_args)
        )
        return obj

    def _get_test_template_class(self, name: str) -> Type[TestTemplate]:
        """
        Dynamically retrieves the appropriate TestTemplate subclass based on the given name.

        Args:
            name (str): The name of the test template.

        Returns:
            Type[TestTemplate]: A subclass of TestTemplate corresponding to the given name.
        """
        template_classes = Registry().test_templates_map

        if name in template_classes:
            return template_classes[name]
        else:
            raise ValueError(f"Unsupported test_template name: {name}")

    def _extract_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract arguments, maintaining their structure, and includes 'values' and 'default' fields where they exist.

        Args:
            args (Dict[str, Any]): The original arguments dictionary.

        Returns:
            Dict[str, Any]: The extracted arguments.
        """
        extracted_args = {}
        for key, value in args.items():
            # Assume nested dictionaries represent complex arguments
            if isinstance(value, dict):
                extracted_args[key] = value
            else:
                extracted_args[key] = value
        return extracted_args

    def _validate_args(self, args: Dict[str, Any], arg_type: str) -> None:
        """
        Validate the extracted arguments against their specified types and constraints.

        Converting and checking default values as necessary.

        Args:
            args (Dict[str, Any]): The arguments to validate.
            arg_type (str): The context of the arguments (e.g., "Environment").

        Raises:
            ValueError: If any argument does not conform to its specified type
            or constraints.
        """
        for arg, details in args.items():
            self._check_and_set_defaults(details, arg, arg_type)

    def _check_and_set_defaults(self, details: Dict[str, Any], arg: str, arg_type: str):
        """
        Check and set default values for arguments based on their type.

        Args:
            details (Dict[str, Any]): Details of the argument including type and default value.
            arg (str): Argument name.
            arg_type (str): Type of argument (e.g., "Environment").

        Raises:
            ValueError: If the argument does not conform to its specified type or constraints.
        """
        if "values" in details and "default" in details and "type" not in details:
            details["type"] = "preset"

        if "type" in details:
            if details["type"] not in self.VALID_DATA_TYPES:
                raise ValueError(f"{arg_type} argument '{arg}' has unsupported type '{details['type']}'.")

            if details["type"] == "bool":
                self._validate_boolean(details, arg, arg_type)
            elif details["type"] in ["int", "str"]:
                self._validate_type(details, arg, arg_type)
            elif details["type"] == "preset":
                self._validate_preset(details, arg, arg_type)
        else:
            for arg, nested_details in details.items():
                self._check_and_set_defaults(nested_details, arg, arg_type)

    def _validate_boolean(self, details: Dict[str, Any], arg: str, arg_type: str):
        if "default" in details:
            if details["default"].lower() == "true":
                converted_value = True
            elif details["default"].lower() == "false":
                converted_value = False
            else:
                raise ValueError(
                    f"{arg_type} argument '{arg}' default value '{details['default']}' is not a valid boolean."
                )
            details["default"] = converted_value

    def _validate_type(self, details: Dict[str, Any], arg: str, arg_type: str):
        if "default" in details:
            try:
                converted_value = eval(details["type"])(details["default"])
            except ValueError as e:
                raise ValueError(
                    f"{arg_type} argument '{arg}' default value "
                    f"'{details['default']}' cannot be converted to type "
                    f"'{details['type']}'."
                ) from e
            details["default"] = converted_value

    def _validate_preset(self, details: Dict[str, Any], arg: str, arg_type: str):
        if details["default"] not in details["values"]:
            raise ValueError(
                f"{arg_type} argument '{arg}' default value '{details['default']}' not in {details['values']}."
            )
