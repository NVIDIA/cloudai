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

import asyncio
import logging
import signal
import sys
from abc import ABC, abstractmethod
from asyncio import Task
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Dict, List, Optional

from .base_job import BaseJob
from .exceptions import JobFailureError, JobSubmissionError
from .job_status_result import JobStatusResult
from .system import System
from .test_scenario import TestRun, TestScenario


class BaseRunner(ABC):
    """
    Abstract base class for a Runner that manages test execution.

    This class provides a framework for executing tests within a given test scenario, handling dependencies and
    execution order.

    Attributes
        mode (str): The operation mode ('dry-run', 'run').
        system (System): The system schema object.
        test_scenario (TestScenario): The test scenario to run.
        output_path (Path): Path to the output directory.
        monitor_interval (int): Interval in seconds for monitoring jobs.
        jobs (List[BaseJob]): List to track jobs created by the runner.
        test_to_job_map (Dict[Test, BaseJob]): Mapping from tests to their jobs.
        logger (logging.Logger): Logger for the runner.
        shutting_down (bool): A flag indicating whether a shutdown process has been initiated, preventing the start of
            new tests and ensuring a graceful termination of all running tests.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario):
        """
        Initialize the BaseRunner with a system object, test scenario, and monitor interval.

        Args:
            mode (str): The operation mode ('dry-run', 'run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.
        """
        self.mode = mode
        self.system = system
        self.test_scenario = test_scenario
        self.output_path = self.setup_output_directory(system.output_path)
        self.monitor_interval = system.monitor_interval
        self.jobs: List[BaseJob] = []
        self.testrun_to_job_map: Dict[TestRun, BaseJob] = {}
        logging.debug(f"{self.__class__.__name__} initialized")
        self.shutting_down = False
        self.register_signal_handlers()

    def setup_output_directory(self, base_output_path: Path) -> Path:
        """
        Set up and return the output directory path for the runner instance.

        Args:
            base_output_path (Path): The base output directory.

        Returns:
            Path: The path to the output directory.
        """
        if not base_output_path.exists():
            base_output_path.mkdir()
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_subpath = base_output_path / f"{self.test_scenario.name}_{current_time}"
        output_subpath.mkdir()
        return output_subpath

    def register_signal_handlers(self):
        """Register signal handlers for handling termination-related signals."""
        signals = [
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGQUIT,
        ]
        for sig in signals:
            signal.signal(sig, self.signal_handler)

    def signal_handler(
        self,
        signum: int,
        frame: Optional[FrameType],  # noqa: Vulture
    ) -> None:
        """
        Respond to termination-related signals (e.g., SIGINT) by initiating a graceful shutdown of the application.

        This method logs the received signal and then triggers the asynchronous shutdown process, which involves
        terminating all outstanding jobs in a controlled manner.

        Args:
            signum (int): The signal number indicating the type of signal received.
            frame (Optional[FrameType]): The current stack frame when the signal was received, or None if not
                applicable. This parameter is typically not used directly but is necessary for signal handler
                functions.

        Returns:
            None
        """
        self.shutting_down = True
        logging.info(f"Signal {signum} received, shutting down...")
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Gracefully shut down the runner, terminating all outstanding jobs."""
        if not self.jobs:
            return
        logging.info("Terminating all jobs...")
        for job in self.jobs:
            self.system.kill(job)
        logging.info("All jobs have been killed.")

        sys.exit(0)

    async def run(self):
        """Asynchronously run the test scenario."""
        if self.shutting_down:
            return

        logging.info("Starting test scenario execution.")
        total_tests = len(self.test_scenario.test_runs)
        completed_jobs_count = 0

        dependency_free_tests = self.find_dependency_free_tests()
        for tr in dependency_free_tests:
            await self.submit_test(tr)

        while completed_jobs_count < total_tests:
            await self.check_start_post_init_dependencies()
            completed_jobs_count += await self.monitor_jobs()
            await asyncio.sleep(self.monitor_interval)

    async def submit_test(self, tr: TestRun):
        """
        Start a dependency-free test.

        Args:
            tr (TestRun): The test to be started.
        """
        logging.info(f"Starting test: {tr.name}")
        try:
            job = self._submit_test(tr)
            self.jobs.append(job)
            self.testrun_to_job_map[tr] = job
        except JobSubmissionError as e:
            logging.error(e)
            exit(1)

    async def delayed_submit_test(self, tr: TestRun, delay: int = 0):
        """
        Delay the start of a test based on start_post_comp dependency.

        Args:
            tr (TestRun): The test to start after a delay.
            delay (int): Delay in seconds before starting the test.
        """
        logging.info(f"Delayed start for test {tr.name} by {delay} seconds.")
        await asyncio.sleep(delay)
        await self.submit_test(tr)

    @abstractmethod
    def _submit_test(self, tr: TestRun) -> BaseJob:
        """
        Execute a given test and returns a job if successful.

        Args:
            tr (TestRun): The test to be executed.

        Returns:
            BaseJob: A BaseJob object
        """
        pass

    async def check_start_post_init_dependencies(self):
        """
        Check and handle start_post_init dependencies.

        This method should be called periodically to ensure timely execution of tests with
        start_post_init dependencies.
        """
        items = list(self.testrun_to_job_map.items())

        for tr, job in items:
            if self.mode == "dry-run" or self.system.is_job_running(job):
                await self.check_and_schedule_start_post_init_dependent_tests(tr)

    async def check_and_schedule_start_post_init_dependent_tests(self, started_test_run: TestRun):
        """
        Schedule tests with a start_post_init dependency on the provided started_test.

        Args:
            started_test_run (TestRun): The test that has just started.
        """
        for tr in self.test_scenario.test_runs:
            if tr not in self.testrun_to_job_map:
                for dep_type, dep in tr.dependencies.items():
                    if (dep_type == "start_post_init") and (dep.test_run == started_test_run):
                        await self.delayed_submit_test(tr)

    def find_dependency_free_tests(self) -> List[TestRun]:
        """
        Find tests that have no 'start_post_comp' or 'start_post_init' dependencies.

        Tests with only 'end_post_comp' dependencies or no dependencies at all are considered dependency-free.

        Returns
            List[Test]: A list of tests that are ready to run without waiting for other tests to start or complete.
        """
        dependency_free_tests = []
        for tr in self.test_scenario.test_runs:
            if "start_post_comp" not in tr.dependencies and "start_post_init" not in tr.dependencies:
                dependency_free_tests.append(tr)

        return dependency_free_tests

    def get_job_output_path(self, tr: TestRun) -> Path:
        """
        Generate and ensure the existence of the output directory for a given test.

        It constructs the path based on the test's section name and current iteration, creating the directories if they
        do not exist.

        Args:
            tr (TestRun): The test run object.

        Returns:
            Path: The path to the job's output directory.

        Raises:
            ValueError: If the test's section name is None.
            FileNotFoundError: If the base output directory does not exist.
            PermissionError: If there is a permission issue creating the directories.
        """
        if not self.output_path.exists():
            raise FileNotFoundError(f"Output directory {self.output_path} does not exist")

        job_output_path = Path()  # avoid reportPossiblyUnboundVariable from pyright

        try:
            test_output_path = self.output_path / tr.name
            test_output_path.mkdir()
            job_output_path = test_output_path / str(tr.current_iteration)
            job_output_path.mkdir()
        except PermissionError as e:
            raise PermissionError(f"Cannot create directory {job_output_path}: {e}") from e

        return job_output_path

    async def monitor_jobs(self) -> int:
        """
        Monitor the status of jobs, handle end_post_comp dependencies, and schedule start_post_comp dependent jobs.

        Returns
            int: The number of completed jobs.
        """
        successful_jobs_count = 0

        for job in list(self.jobs):
            if self.mode == "dry-run" or self.system.is_job_completed(job):
                await self.job_completion_callback(job)

                if self.mode == "dry-run":
                    successful_jobs_count += 1
                    await self.handle_job_completion(job)
                else:
                    if self.test_scenario.job_status_check:
                        job_status_result = self.get_job_status(job)
                        if job_status_result.is_successful:
                            successful_jobs_count += 1
                            await self.handle_job_completion(job)
                        else:
                            error_message = (
                                f"Job {job.id} for test {job.test_run.name} failed: "
                                f"{job_status_result.error_message}"
                            )
                            logging.error(error_message)
                            await self.shutdown()
                            raise JobFailureError(job.test_run.name, error_message, job_status_result.error_message)
                    else:
                        job_status_result = self.get_job_status(job)
                        if not job_status_result.is_successful:
                            error_message = (
                                f"Job {job.id} for test {job.test_run.name} failed: "
                                f"{job_status_result.error_message}"
                            )
                            logging.error(error_message)
                        successful_jobs_count += 1
                        await self.handle_job_completion(job)

        return successful_jobs_count

    def get_job_status(self, job: BaseJob) -> JobStatusResult:
        """
        Retrieve the job status from a specified output directory.

        Args:
            job (BaseJob): The job to be checked.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        return job.test_run.test.test_template.get_job_status(job.test_run.output_path)

    async def handle_job_completion(self, completed_job: BaseJob):
        """
        Handle the completion of a job, including dependency management and iteration control.

        Args:
            completed_job (BaseJob): The job that has just been completed.
        """
        logging.info(f"Job completed: {completed_job.test_run.name}")

        self.jobs.remove(completed_job)
        del self.testrun_to_job_map[completed_job.test_run]
        completed_job.test_run.current_iteration += 1
        if not completed_job.terminated_by_dependency and completed_job.test_run.has_more_iterations():
            msg = f"Re-running job for iteration {completed_job.test_run.current_iteration}"
            logging.info(msg)
            await self.submit_test(completed_job.test_run)
        else:
            await self.handle_dependencies(completed_job)

    async def job_completion_callback(self, job: BaseJob) -> None:  # noqa: B027
        """
        Call callback functions upon job completion.

        This method can be overridden by subclasses to invoke custom actions or callback functions when a job
        completes, such as storing logs or other processing.

        Args:
            job (BaseJob): The job that has completed and for which callback functions are being invoked.
        """
        pass

    async def handle_dependencies(self, completed_job: BaseJob) -> List[Task]:
        """
        Handle the start_post_comp and end_post_comp dependencies for a completed job.

        Args:
            completed_job (BaseJob): The job that has just been completed.

        Returns:
            List[asyncio.Task]: A list of asyncio.Task objects created for handling the dependencies.
        """
        tasks = []

        # Handling start_post_comp dependencies
        for tr in self.test_scenario.test_runs:
            if tr not in self.testrun_to_job_map:
                for dep_type, dep in tr.dependencies.items():
                    if dep_type == "start_post_comp" and dep.test_run.test == completed_job.test_run.test:
                        task = await self.delayed_submit_test(tr)
                        if task:
                            tasks.append(task)

        # Handling end_post_comp dependencies
        for test, dependent_job in self.testrun_to_job_map.items():
            for dep_type, dep in test.dependencies.items():
                if dep_type == "end_post_comp" and dep.test_run.test == completed_job.test_run.test:
                    task = await self.delayed_kill_job(dependent_job)
                    tasks.append(task)

        return tasks

    async def delayed_kill_job(self, job: BaseJob, delay: int = 0):
        """
        Schedule termination of a Standalone job after a specified delay.

        Args:
            job (BaseJob): The job to be terminated.
            delay (int): Delay in seconds after which the job should be terminated.
        """
        logging.info(f"Scheduling termination of job {job.id} after {delay} seconds.")
        await asyncio.sleep(delay)
        job.terminated_by_dependency = True
        self.system.kill(job)
