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

from typing import Dict, List, Union

from ..models.workload import TestDefinition
from .test_template import TestTemplate


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

    @property
    def name(self) -> str:
        return self.test_definition.name

    @property
    def description(self) -> str:
        return self.test_definition.description

    @property
    def cmd_args(self) -> Dict[str, Union[str, List[str]]]:
        return self.test_definition.cmd_args_dict

    @property
    def extra_cmd_args(self) -> str:
        return self.test_definition.extra_args_str

    @property
    def extra_env_vars(self) -> Dict[str, Union[str, List[str]]]:
        return self.test_definition.extra_env_vars
