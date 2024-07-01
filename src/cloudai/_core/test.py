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

import sys
from typing import Dict, List, Optional, Union

from .job_status_result import JobStatusResult
from .test_template import TestTemplate


class Test:
    """
    Represent a test, an instance of a test template with custom arguments, node configuration, and other details.

    Attributes
        name (str): Unique name of the test.
        description (str): Description of the test.
        test_template (TestTemplate): The test template object.
        env_vars (Dict[str, str]): Environment variables for the test.
        cmd_args (Dict[str, str]): Arguments for the test.
        extra_env_vars (Dict[str, str]): Extra environment variables.
        extra_cmd_args (str): Extra command-line arguments.
        section_name (str): The section name of the test in the configuration.
        dependencies (Optional[Dict[str, Optional['TestDependency']]]): Dependencies of the test.
        iterations (Union[int, str]): Number of iterations to run the test.
        current_iteration (int): The current iteration count.
        num_nodes (int): The number of nodes to be used for the test execution.
        nodes (List[str]): List of nodes involved in the test.
        sol (Optional[float]): Speed-of-light performance for reference.
        weight (float): The weight of this test in a test scenario, indicating its relative importance or priority.
        ideal_perf (float): The ideal performance value for comparison.
        time_limit (Optional[str]): Time limit for the test specified as a string in "hh:mm:ss" format, or None if no
            limit.
    """

    __test__ = False

    def __init__(
        self,
        name: str,
        description: str,
        test_template: TestTemplate,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        section_name: str = "",
        dependencies: Optional[Dict[str, "TestDependency"]] = None,
        iterations: Union[int, str] = 1,
        num_nodes: int = 1,
        nodes: Optional[List[str]] = None,
        sol: Optional[float] = None,
        weight: float = 0.0,
        ideal_perf: float = 1.0,
        time_limit: Optional[str] = None,
    ) -> None:
        """
        Initialize a Test instance.

        Args:
            name (str): Name of the test.
            description (str): Description of the test.
            test_template (TestTemplate): Test template object.
            env_vars (Dict[str, str]): Environment variables for the test.
            cmd_args (Dict[str, str]): Command-line arguments for the test.
            extra_env_vars (Dict[str, str]): Extra environment variables.
            extra_cmd_args (str): Extra command-line arguments.
            section_name (str): The section name of the test in the configuration.
            dependencies (Optional[Dict[str, TestDependency]]): Test dependencies.
            iterations (Union[int, str]): Total number of iterations to run the test. Can be an integer or 'infinite'
                for endless iterations.
            num_nodes (int): The number of nodes to be used for the test execution.
            nodes (List[str]): List of nodes to be used in the test.
            sol (Optional[float]): Speed-of-light performance for reference.
            weight (float): The weight of this test in a test scenario, indicating its relative importance or priority.
            ideal_perf (float): The ideal performance value for comparison.
            time_limit (Optional[str]): Time limit for the test specified as a string
        """
        self.name = name
        self.description = description
        self.test_template = test_template
        self.env_vars = env_vars
        self.cmd_args = cmd_args
        self.extra_env_vars = extra_env_vars
        self.extra_cmd_args = extra_cmd_args
        self.section_name = section_name
        self.dependencies = dependencies or {}
        self.iterations = iterations if isinstance(iterations, int) else sys.maxsize
        self.current_iteration = 0
        self.num_nodes = num_nodes
        self.nodes = nodes if nodes else []
        self.sol = sol
        self.weight = weight
        self.ideal_perf = ideal_perf
        self.time_limit = time_limit

    def __repr__(self) -> str:
        """
        Return a string representation of the Test instance.

        Returns
            str: String representation of the test.
        """
        return (
            f"Test(name={self.name}, description={self.description}, "
            f"test_template={self.test_template.name}, "
            f"env_vars={self.env_vars}, "
            f"cmd_args={self.cmd_args}, "
            f"extra_env_vars={self.extra_env_vars}, "
            f"extra_cmd_args={self.extra_cmd_args}, "
            f"section_name={self.section_name}, "
            f"dependencies={self.dependencies}, iterations={self.iterations}, "
            f"nodes={self.nodes})"
        )

    def gen_exec_command(self, output_path: str) -> str:
        """
        Generate the command to run this specific test.

        Args:
            output_path (str): Path to the output directory.

        Returns:
            str: The command string.
        """
        if self.time_limit is not None:
            self.cmd_args["time_limit"] = self.time_limit

        return self.test_template.gen_exec_command(
            self.env_vars,
            self.cmd_args,
            self.extra_env_vars,
            self.extra_cmd_args,
            output_path,
            self.num_nodes,
            self.nodes,
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

    def get_job_status(self, output_path: str) -> JobStatusResult:
        """
        Determine the status of a job based on the outputs located in the given output directory.

        Args:
            output_path (str): Path to the output directory.

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
