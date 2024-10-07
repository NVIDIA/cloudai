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

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from .test_template import TestTemplate


class Test:
    """Represent a test, an instance of a test template with custom arguments, node configuration, and other details."""

    __test__ = False

    def __init__(
        self,
        name: str,
        description: str,
        test_template: TestTemplate,
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        sol: Optional[float] = None,
        weight: float = 0.0,
        ideal_perf: float = 1.0,
    ) -> None:
        """
        Initialize a Test instance.

        Args:
            name (str): Name of the test.
            description (str): Description of the test.
            test_template (TestTemplate): Test template object.
            cmd_args (Dict[str, str]): Command-line arguments for the test.
            extra_env_vars (Dict[str, str]): Extra environment variables.
            extra_cmd_args (str): Extra command-line arguments.
            iterations (Union[int, str]): Total number of iterations to run the test. Can be an integer or 'infinite'
                for endless iterations.
            sol (Optional[float]): Speed-of-light performance for reference.
            weight (float): The weight of this test in a test scenario, indicating its relative importance or priority.
            ideal_perf (float): The ideal performance value for comparison.
        """
        self.name = name
        self.description = description
        self.test_template = test_template
        self.cmd_args = cmd_args
        self.extra_env_vars = extra_env_vars
        self.extra_cmd_args = extra_cmd_args
        self.sol = sol
        self.weight = weight
        self.ideal_perf = ideal_perf

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
