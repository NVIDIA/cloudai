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

from typing import Any, Dict, Type

from cloudai.schema.core import System, TestTemplate

from .base_multi_file_parser import BaseMultiFileParser


class TestTemplateParser(BaseMultiFileParser):
    """
    Parser for creating TestTemplate objects from provided data.

    Attributes:
        system (System): The system schema object.
    """

    VALID_DATA_TYPES = ["preset", "bool", "int", "str"]

    def __init__(self, system: System, directory_path: str) -> None:
        """
        Initializes a TestTemplateParser with a specific system and directory path.

        Args:
            system (System): The system schema object.
            directory_path (str): The directory path where test templates are located.
        """
        super().__init__(directory_path)
        self.system = system
        self.directory_path: str = directory_path

    def _parse_data(self, data: Dict[str, Any]) -> TestTemplate:
        """
        Parses data for a TestTemplate object.

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

        return test_template_class(system=self.system, name=name, env_vars=env_vars, cmd_args=cmd_args)

    def _get_test_template_class(self, name: str) -> Type[TestTemplate]:
        """
        Dynamically retrieves the appropriate TestTemplate subclass based on the given name.

        Args:
            name (str): The name of the test template.

        Returns:
            Type[TestTemplate]: A subclass of TestTemplate corresponding to the given name.
        """
        template_classes = self._enumerate_test_template_classes()

        if name in template_classes:
            return template_classes[name]
        else:
            raise ValueError(f"Unsupported test_template name: {name}")

    @staticmethod
    def _enumerate_test_template_classes() -> Dict[str, Type[TestTemplate]]:
        """
        Dynamically enumerates all subclasses of TestTemplate available in the
        current namespace and maps their class names to the class objects.

        Returns:
            Dict[str, Type[TestTemplate]]: A dictionary mapping class names to
            TestTemplate subclasses.
        """
        return {cls.__name__: cls for cls in TestTemplate.__subclasses__()}

    def _extract_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts arguments, maintaining their structure, and includes 'values'
        and 'default' fields where they exist.

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
        Validates the extracted arguments against their specified types and
        constraints, converting and checking default values as necessary.

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
        Helper method to check and set default values for arguments based on their type.

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
