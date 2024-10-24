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

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from .test_template import TestTemplate


class Installable(ABC):
    """Installable object."""

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...


class Test:
    """Represent a test, an instance of a test template with custom arguments, node configuration, and other details."""

    __test__ = False

    def __init__(self, test_definition: "TestDefinition", test_template: TestTemplate) -> None:
        """
        Initialize a Test instance.

        Args:
            test_definition (TestDefinition): The test definition object.
            test_template (TestTemplate): The test template object
        """
        self.test_template = test_template
        self.test_definition = test_definition

    def __repr__(self) -> str:
        """
        Return a string representation of the Test instance.

        Returns
            str: String representation of the test.
        """
        return (
            f"Test(name={self.name}, description={self.description}, "
            f"test_template={self.test_template.name}, "
            f"cmd_args={self.cmd_args}, "
            f"extra_env_vars={self.extra_env_vars}, "
            f"extra_cmd_args={self.extra_cmd_args}"
        )

    @property
    def name(self) -> str:
        return self.test_definition.name

    @property
    def description(self) -> str:
        return self.test_definition.description

    @property
    def cmd_args(self) -> Dict[str, str]:
        return self.test_definition.cmd_args_dict

    @property
    def extra_cmd_args(self) -> str:
        return self.test_definition.extra_args_str

    @property
    def extra_env_vars(self) -> Dict[str, str]:
        return self.test_definition.extra_env_vars


class CmdArgs(BaseModel):
    """Test command arguments."""

    model_config = ConfigDict(extra="forbid")


class TestDefinition(BaseModel):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    extra_env_vars: dict[str, str] = {}
    extra_cmd_args: dict[str, str] = {}

    @property
    def cmd_args_dict(self) -> Dict[str, str]:
        return self.cmd_args.model_dump()

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k}={v}" if v else k)
        return " ".join(parts)

    @property
    @abstractmethod
    def installables(self) -> list[Installable]:
        return []
