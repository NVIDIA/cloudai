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

from pathlib import Path
from typing import Any, Dict, Optional

from .command_gen_strategy import CommandGenStrategy
from .grading_strategy import GradingStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .job_status_result import JobStatusResult
from .job_status_retrieval_strategy import JobStatusRetrievalStrategy
from .json_gen_strategy import JsonGenStrategy
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test_scenario import TestRun


class TestTemplate:
    """
    Base class representing a test template.

    Providing a framework for test execution, including installation, uninstallation, and execution command generation
    based on system configurations and test parameters.

    Attributes
        name (str): Unique name of the test template.
        cmd_args (Dict[str, Any]): Default command-line arguments.
        logger (logging.Logger): Logger for the test template.
        install_strategy (InstallStrategy): Strategy for installing test prerequisites.
        command_gen_strategy (CommandGenStrategy): Strategy for generating execution commands.
        json_gen_strategy (JsonGenStrategy): Strategy for generating json string.
        job_id_retrieval_strategy (JobIdRetrievalStrategy): Strategy for retrieving job IDs.
        report_generation_strategy (ReportGenerationStrategy): Strategy for generating reports.
        grading_strategy (GradingStrategy): Strategy for grading performance based on test outcomes.
        job_status_retrieval_strategy (JobStatusRetrievalStrategy): Strategy for determining job statuses.
    """

    __test__ = False

    def __init__(self, system: System, name: str) -> None:
        """
        Initialize a TestTemplate instance.

        Args:
            system (System): System configuration for the test template.
            name (str): Name of the test template.
            cmd_args (Dict[str, Any]): Command-line arguments.
        """
        self.system = system
        self.name = name
        self.command_gen_strategy: Optional[CommandGenStrategy] = None
        self.json_gen_strategy: Optional[JsonGenStrategy] = None
        self.job_id_retrieval_strategy: Optional[JobIdRetrievalStrategy] = None
        self.job_status_retrieval_strategy: Optional[JobStatusRetrievalStrategy] = None
        self.report_generation_strategy: Optional[ReportGenerationStrategy] = None
        self.grading_strategy: Optional[GradingStrategy] = None

    def __repr__(self) -> str:
        """
        Return a string representation of the TestTemplate instance.

        Returns
            str: String representation of the test template.
        """
        return f"TestTemplate(name={self.name})"

    def gen_exec_command(self, tr: TestRun) -> str:
        """
        Generate an execution command for a test using this template.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated execution command.
        """
        if self.command_gen_strategy is None:
            raise ValueError(
                "command_gen_strategy is missing. Ensure the strategy is registered in the Registry "
                "by calling the appropriate registration function for the system type."
            )
        return self.command_gen_strategy.gen_exec_command(tr)

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        """
        Generate a JSON string representing the Kubernetes job specification for this test using this template.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            Dict[Any, Any]: A dictionary representing the Kubernetes job specification.
        """
        if self.json_gen_strategy is None:
            raise ValueError(
                "json_gen_strategy is missing. Ensure the strategy is registered in the Registry "
                "by calling the appropriate registration function for the system type."
            )
        return self.json_gen_strategy.gen_json(tr)

    def get_job_id(self, stdout: str, stderr: str) -> Optional[int]:
        """
        Retrieve the job ID from the execution output using the job ID retrieval strategy.

        Args:
            stdout (str): Standard output from the test execution.
            stderr (str): Standard error from the test execution.

        Returns:
            Optional[int]: The retrieved job ID, or None if not found.
        """
        if self.job_id_retrieval_strategy is None:
            raise ValueError(
                "job_id_retrieval_strategy is missing. Ensure the strategy is registered in the Registry "
                "by calling the appropriate registration function for the system type."
            )
        return self.job_id_retrieval_strategy.get_job_id(stdout, stderr)

    def get_job_status(self, output_path: Path) -> JobStatusResult:
        """
        Determine the job status by evaluating the contents or results in a specified output directory.

        Args:
            output_path (Path): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        if self.job_status_retrieval_strategy is None:
            raise ValueError(
                "job_status_retrieval_strategy is missing. Ensure the strategy is registered in "
                "the Registry by calling the appropriate registration function for the system type."
            )
        return self.job_status_retrieval_strategy.get_job_status(output_path)

    def can_handle_directory(self, directory_path: Path) -> bool:
        """
        Determine if the strategy can handle the directory.

        Args:
            directory_path (Path): Path to the directory.

        Returns:
            bool: True if can handle, False otherwise.
        """
        if self.report_generation_strategy is not None:
            return self.report_generation_strategy.can_handle_directory(directory_path)
        else:
            return False

    def generate_report(self, test_name: str, directory_path: Path, sol: Optional[float] = None) -> None:
        """
        Generate a report from the directory.

        Args:
            test_name (str): The name of the test.
            directory_path (Path): Path to the directory.
            sol (Optional[float]): Speed-of-light performance for reference.
        """
        if self.report_generation_strategy is not None:
            return self.report_generation_strategy.generate_report(test_name, directory_path, sol)

    def grade(self, directory_path: Path, ideal_perf: float) -> Optional[float]:
        """
        Read the performance value from the directory.

        Args:
            directory_path (Path): Path to the directory containing performance data.
            ideal_perf (float): The ideal performance metric to compare against.

        Returns:
            Optional[float]: The performance value read from the directory.
        """
        if self.grading_strategy is not None:
            return self.grading_strategy.grade(directory_path, ideal_perf)
