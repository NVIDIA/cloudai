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

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .job_status_result import JobStatusResult
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
        section_name: str = "",
        dependencies: Optional[Dict[str, "TestDependency"]] = None,
        iterations: Union[int, str] = 1,
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
            section_name (str): The section name of the test in the configuration.
            dependencies (Optional[Dict[str, TestDependency]]): Test dependencies.
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
        self.section_name = section_name
        self.dependencies = dependencies or {}
        self.iterations = iterations if isinstance(iterations, int) else sys.maxsize
        self.current_iteration = 0
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
            f"extra_cmd_args={self.extra_cmd_args}, "
            f"section_name={self.section_name}, "
            f"dependencies={self.dependencies}, iterations={self.iterations}, "
        )

    def gen_exec_command(
        self, output_path: Path, time_limit: Optional[str] = None, num_nodes: int = 1, nodes: Optional[List[str]] = None
    ) -> str:
        """
        Generate the command to run this specific test.

        Args:
            output_path (Path): Path to the output directory where logs and results will be stored.
            time_limit (Optional[str]): Time limit for the test execution.
            num_nodes (Optional[int]): Number of nodes to be used for the test execution.
            nodes (Optional[List[str]]): List of nodes involved in the test.

        Returns:
            str: The command string.
        """
        if time_limit is not None:
            self.cmd_args["time_limit"] = time_limit
        if not nodes:
            nodes = []

        return self.test_template.gen_exec_command(
            self.cmd_args,
            self.extra_env_vars,
            self.extra_cmd_args,
            output_path,
            num_nodes,
            nodes,
        )

    def gen_json(
        self,
        output_path: Path,
        job_name: str,
        time_limit: Optional[str] = None,
        num_nodes: int = 1,
        nodes: Optional[List[str]] = None,
    ) -> Dict[Any, Any]:
        """
        Generate a JSON dictionary representing the Kubernetes job specification for this test.

        Args:
            output_path (Path): Path to the output directory where logs and results will be stored.
            job_name (str): The name assigned to the Kubernetes job.
            time_limit (Optional[str]): Time limit for the test execution.
            num_nodes (Optional[int]): Number of nodes to be used for the test execution.
            nodes (Optional[List[str]]): List of nodes involved in the test.

        Returns:
            Dict[Any, Any]: A dictionary representing the Kubernetes job specification.
        """
        if time_limit is not None:
            self.cmd_args["time_limit"] = time_limit
        if not nodes:
            nodes = []

        return self.test_template.gen_json(
            self.cmd_args,
            self.extra_env_vars,
            self.extra_cmd_args,
            output_path,
            job_name,
            num_nodes,
            nodes,
        )

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Retrieve the job ID using the test template's method.

        Args:
            stdout (str): Standard output from the command execution.
            stderr (str): Standard error from the command execution.

        Returns:
            Optional[int]: The retrieved job ID, or None if not found.
        """
        return self.test_template.get_job_id(stdout, stderr)

    def get_job_status(self, output_path: Path) -> JobStatusResult:
        """
        Determine the status of a job based on the outputs located in the given output directory.

        Args:
            output_path (Path): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        return self.test_template.get_job_status(output_path)

    def has_more_iterations(self) -> bool:
        """
        Check if the test has more iterations to run.

        Returns
            bool: True if more iterations are pending, False otherwise.
        """
        return self.current_iteration < self.iterations


class TestDependency:
    """
    Represents a dependency for a test.

    Attributes
        test (Test): The test object it depends on.
        time (int): Time in seconds after which this dependency is met.
    """

    __test__ = False

    def __init__(self, test: Test, time: int) -> None:
        """
        Initialize a TestDependency instance.

        Args:
            test (Test): The test object it depends on.
            time (int): Time in seconds to meet the dependency.
        """
        self.test = test
        self.time = time


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
