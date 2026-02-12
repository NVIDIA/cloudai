# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from .base_job import BaseJob
from .command_gen_strategy import CommandGenStrategy
from .exceptions import JobFailureError, JobSubmissionError
from .job_status_result import JobStatusResult
from .json_gen_strategy import JsonGenStrategy
from .registry import Registry
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

    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path):
        """
        Initialize the BaseRunner with a system object, test scenario, and monitor interval.

        Args:
            mode (str): The operation mode ('dry-run', 'run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.
            output_path (Path): Path to the output directory.
        """
        self.mode = mode
        self.system = system
        self.test_scenario = test_scenario
        self.scenario_root = output_path
        self.monitor_interval = system.monitor_interval
        self.jobs: List[BaseJob] = []
        self.testrun_to_job_map: Dict[TestRun, BaseJob] = {}
        logging.debug(f"{self.__class__.__name__} initialized")
        self.shutting_down = False

    def shutdown(self):
        """Gracefully shut down the runner, terminating all outstanding jobs."""
        self.shutting_down = True
        logging.info("Terminating all jobs...")
        for job in self.jobs:
            logging.info(f"Terminating job {job.id} for test {job.test_run.name}")
            self.system.kill(job)
        logging.info("Waiting for all jobs to be killed.")

    def run(self):
        """Run the test scenario."""
        if self.shutting_down:
            return

        total_tests = len(self.test_scenario.test_runs)
        dependency_free_trs = self.find_dependency_free_tests()
        for tr in dependency_free_trs:
            self.submit_test(tr)

        logging.debug(f"Total tests: {total_tests}, dependency free tests: {[tr.name for tr in dependency_free_trs]}")
        while self.jobs:
            self.check_start_post_init_dependencies()
            self.monitor_jobs()
            logging.debug(f"sleeping for {self.monitor_interval} seconds")
            time.sleep(self.monitor_interval)

    def submit_test(self, tr: TestRun):
        """
        Start a dependency-free test.

        Args:
            tr (TestRun): The test to be started.
        """
        tr.output_path = self.get_job_output_path(tr)
        logging.info(f"Starting test: {tr.name} (results at: {tr.output_path})")
        self.on_job_submit(tr)
        try:
            job = self._submit_test(tr)
            self.jobs.append(job)
            self.testrun_to_job_map[tr] = job
        except JobSubmissionError as e:
            logging.error(e)
            exit(1)

    def on_job_submit(self, tr: TestRun) -> None:
        return

    def delayed_submit_test(self, tr: TestRun, delay: int = 5):
        """
        Delay the start of a test based on start_post_comp dependency.

        Args:
            tr (TestRun): The test to start after a delay.
            delay (int): Delay in seconds before starting the test.
        """
        logging.debug(f"Delayed start for test {tr.name} by {delay} seconds.")
        time.sleep(delay)
        self.submit_test(tr)

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

    def check_start_post_init_dependencies(self):
        """
        Check and handle start_post_init dependencies.

        This method should be called periodically to ensure timely execution of tests with
        start_post_init dependencies.
        """
        items = list(self.testrun_to_job_map.items())

        for tr, job in items:
            is_running, is_completed = False, False
            if self.mode == "dry-run":
                is_running, is_completed = True, True
            else:
                is_running, is_completed = (
                    self.system.is_job_running(job),
                    self.system.is_job_completed(job),
                )

            logging.debug(f"start_post_init for test {tr.name} ({is_running=}, {is_completed=}, {self.mode=})")
            if is_running or is_completed:
                self.check_and_schedule_start_post_init_dependent_tests(tr)

    def check_and_schedule_start_post_init_dependent_tests(self, started_test_run: TestRun):
        """
        Schedule tests with a start_post_init dependency on the provided started_test.

        Args:
            started_test_run (TestRun): The test that has just started.
        """
        for tr in self.test_scenario.test_runs:
            if tr not in self.testrun_to_job_map:
                for dep_type, dep in tr.dependencies.items():
                    if (dep_type == "start_post_init") and (dep.test_run == started_test_run):
                        self.delayed_submit_test(tr)

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

        It constructs the path based on the test's section name and current iteration,
        creating the directories if they do not exist.

        Args:
            tr (TestRun): The test run object.

        Returns:
            Path: The path to the job's output directory.

        Raises:
            ValueError: If the test's section name is None.
            FileNotFoundError: If the base output directory does not exist.
            PermissionError: If there is a permission issue creating the directories.
        """
        if not self.scenario_root.exists():
            self.scenario_root.mkdir()

        job_output_path = self.scenario_root / tr.name / str(tr.current_iteration)
        # here it is required to check DSE as step number because test_definition object is not a DSE object anymore
        if tr.step > 0:
            job_output_path = job_output_path / str(tr.step)

        if not job_output_path.exists():
            try:
                job_output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(f"Cannot create directory {job_output_path}: {e}") from e

        return job_output_path

    def monitor_jobs(self) -> int:
        """
        Monitor the status of jobs, handle end_post_comp dependencies, and schedule start_post_comp dependent jobs.

        Returns
            int: The number of completed jobs.
        """
        successful_jobs_count = 0

        logging.debug(f"Monitoring {len(self.jobs)} jobs")
        for job in list(self.jobs):
            is_completed = True if self.mode == "dry-run" else self.system.is_job_completed(job)

            if is_completed:
                logging.debug(f"Job {job.id} for test {job.test_run.name} completed ({self.mode=}, {is_completed=})")
                self.on_job_completion(job)

                if self.mode == "dry-run":
                    successful_jobs_count += 1
                    self.handle_job_completion(job)
                else:
                    if self.test_scenario.job_status_check:
                        job_status_result = self.get_job_status(job)
                        if job_status_result.is_successful:
                            successful_jobs_count += 1
                            self.handle_job_completion(job)
                        else:
                            error_message = (
                                f"Job {job.id} for test {job.test_run.name} failed: {job_status_result.error_message}"
                            )
                            logging.error(error_message)
                            self.handle_job_completion(job)
                            self.shutdown()
                            raise JobFailureError(job.test_run.name, error_message, job_status_result.error_message)
                    else:
                        job_status_result = self.get_job_status(job)
                        if not job_status_result.is_successful:
                            error_message = (
                                f"Job {job.id} for test {job.test_run.name} failed: {job_status_result.error_message}"
                            )
                            logging.error(error_message)
                        successful_jobs_count += 1
                        self.handle_job_completion(job)

        return successful_jobs_count

    def get_runner_job_status(self, job: BaseJob) -> JobStatusResult:
        return JobStatusResult(is_successful=True)

    def get_job_status(self, job: BaseJob) -> JobStatusResult:
        """
        Retrieve the job status from a specified output directory.

        Args:
            job (BaseJob): The job to be checked.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        runner_job_status_result = self.get_runner_job_status(job)
        workload_run_results = job.test_run.test.was_run_successful(job.test_run)
        if not runner_job_status_result.is_successful:
            return runner_job_status_result
        if not workload_run_results.is_successful:
            return workload_run_results
        return JobStatusResult(is_successful=True)

    def handle_job_completion(self, completed_job: BaseJob):
        """
        Handle the completion of a job, including dependency management and iteration control.

        Args:
            completed_job (BaseJob): The job that has just been completed.
        """
        logging.info(
            f"Job completed: {completed_job.test_run.name} "
            f"(iteration {completed_job.test_run.current_iteration + 1} of {completed_job.test_run.iterations})"
        )

        self.jobs.remove(completed_job)
        del self.testrun_to_job_map[completed_job.test_run]

        if completed_job.test_run.step <= 0:
            if not completed_job.terminated_by_dependency and completed_job.test_run.has_more_iterations():
                completed_job.test_run.current_iteration += 1
                msg = f"Re-running job for iteration {completed_job.test_run.current_iteration}"
                logging.info(msg)
                self.submit_test(completed_job.test_run)
            else:
                self.handle_dependencies(completed_job)

    def on_job_completion(self, job: BaseJob) -> None:
        """
        Call callback functions upon job completion.

        This method can be overridden by subclasses to invoke custom actions or callback functions when a job
        completes, such as storing logs or other processing.

        Args:
            job (BaseJob): The job that has completed and for which callback functions are being invoked.
        """
        return

    def handle_dependencies(self, completed_job: BaseJob):
        """
        Handle the start_post_comp and end_post_comp dependencies for a completed job.

        Args:
            completed_job (BaseJob): The job that has just been completed.
        """
        # Handling start_post_comp dependencies
        for tr in self.test_scenario.test_runs:
            if tr not in self.testrun_to_job_map:
                for dep_type, dep in tr.dependencies.items():
                    if dep_type == "start_post_comp" and dep.test_run == completed_job.test_run:
                        self.delayed_submit_test(tr)

        # Handling end_post_comp dependencies
        for test, dependent_job in self.testrun_to_job_map.items():
            for dep_type, dep in test.dependencies.items():
                if dep_type == "end_post_comp" and dep.test_run == completed_job.test_run:
                    self.delayed_kill_job(dependent_job)

    def delayed_kill_job(self, job: BaseJob, delay: int = 0):
        """
        Schedule termination of a Standalone job after a specified delay.

        Args:
            job (BaseJob): The job to be terminated.
            delay (int): Delay in seconds after which the job should be terminated.
        """
        logging.info(f"Scheduling termination of job {job.id} after {delay} seconds.")
        time.sleep(delay)
        job.terminated_by_dependency = True
        self.system.kill(job)

    def get_cmd_gen_strategy(self, system: System, test_run: TestRun) -> CommandGenStrategy:
        strategy_cls = Registry().get_command_gen_strategy(type(system), type(test_run.test))
        return strategy_cls(system, test_run)

    def get_json_gen_strategy(self, system: System, test_run: TestRun) -> JsonGenStrategy:
        strategy_cls = Registry().get_json_gen_strategy(type(system), type(test_run.test))
        return strategy_cls(system, test_run)
