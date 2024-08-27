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

from typing import Any, Dict

from .system import System


class TestTemplateStrategy:
    """
    Abstract base class representing a test template.

    Provides a framework for test execution, including installation, uninstallation, and execution command
    generation based on system configurations and test parameters.

    Attributes
        system (System): The system schema object.
        install_path (str): Path where the benchmarks are to be installed.
        env_vars (Dict[str, Any]): Default environment variables.
        cmd_args (Dict[str, Any]): Default command-line arguments.
        default_env_vars (Dict[str, str]): Constructed default environment variables.
        default_cmd_args (Dict[str, str]): Constructed default command-line arguments.
    """

    __test__ = False

    def __init__(self, system: System, env_vars: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        """
        Initialize a TestTemplateStrategy instance with system configuration, env variables, and command-line arguments.

        Args:
            system (System): The system configuration for the test.
            env_vars (Dict[str, Any]): Default environment variables.
            cmd_args (Dict[str, Any]): Default command-line arguments.
        """
        self.system = system
        self.install_path = ""
        self.env_vars = env_vars
        self.cmd_args = cmd_args
        self.default_env_vars = self._construct_default_env_vars()
        self.default_cmd_args = self._construct_default_cmd_args()

    def _construct_default_env_vars(self) -> Dict[str, str]:
        """
        Construct the default environment variables for the test template.

        Returns
            Dict[str, str]: A dictionary containing the default environment variables.
        """
        return {
            key: value["default"]
            for key, value in self.env_vars.items()
            if isinstance(value, dict) and "default" in value
        }

    def _construct_default_cmd_args(self) -> Dict[str, str]:
        """
        Construct the default arguments for the test template recursively.

        Returns
            Dict[str, Any]: A dictionary containing the combined default arguments.
        """

        def construct_args(cmd_args: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            args = {}
            for key, value in cmd_args.items():
                full_key = f"{parent_key}.{key}" if parent_key else key

                if isinstance(value, dict):
                    # If 'default' is present, add it to the arguments
                    if "default" in value:
                        args[full_key] = value["default"]

                    # Recursively process nested dictionaries
                    nested_args = construct_args(
                        {k: v for k, v in value.items() if k not in ["type", "default", "values"]},
                        full_key,
                    )
                    args.update(nested_args)
                else:
                    args[full_key] = value
            return args

        return construct_args(self.cmd_args)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten a nested dictionary into a single level dictionary with dot-separated keys.

        Args:
            d (Dict[str, Any]): The dictionary to flatten.
            parent_key (str): The base key for recursion (used internally).
            sep (str): Separator used between keys.

        Returns:
            Dict[str, Any]: Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _override_env_vars(
        self,
        default_env_vars: Dict[str, str],
        provided_env_vars: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Override the default environment variables with provided values.

        Args:
            default_env_vars (Dict[str, str]): The default environment variables.
            provided_env_vars (Dict[str, str]): The provided environment variables to override defaults.

        Returns:
            Dict[str, str]: A dictionary of environment variables with overrides applied.
        """
        final_env_vars = default_env_vars.copy()
        final_env_vars.update(provided_env_vars)
        return final_env_vars

    def _override_cmd_args(
        self,
        default_cmd_args: Dict[str, str],
        provided_cmd_args: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Override the default command-line arguments with provided values.

        Args:
            default_cmd_args (Dict[str, str]): The default command-line arguments.
            provided_cmd_args (Dict[str, str]): The provided command-line arguments to override defaults.

        Returns:
            Dict[str, str]: A dictionary of command-line arguments with overrides applied.
        """
        final_cmd_args = default_cmd_args.copy()
        flattened_args = self._flatten_dict(provided_cmd_args)

        for key, value in flattened_args.items():
            final_cmd_args[key] = value

        return final_cmd_args
