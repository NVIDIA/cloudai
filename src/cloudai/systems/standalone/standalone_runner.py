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

import datetime
import logging
from pathlib import Path
from typing import cast

import cloudai.metrics
from cloudai import output
from cloudai.core import BaseJob, BaseRunner, JobIdRetrievalError, JobStatusResult, System, TestRun, TestScenario
from cloudai.models import output as output_models
from cloudai.util import CommandShell

from .standalone_job import StandaloneJob


class StandaloneRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Standalone.

    Attributes
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path) -> None:
        super().__init__(mode, system, test_scenario, output_path)
        self.cmd_shell = CommandShell()

    def create_experiment_output(self) -> output.ExperimentOutput | None:
        if self.mode != "run":
            return None
        output_path = self.scenario_root.absolute()
        experiment = output_models.Experiment(
            id=output_path.name,
            name=self.test_scenario.name,
            status="running",
            path=str(output_path),
            start=datetime.datetime.now(datetime.timezone.utc),
            tests=[
                output_models.Test(
                    id=str(tr.name),
                    name=tr.test.name,
                    description=tr.test.description,
                    path=str(output_path / str(tr.name)),
                )
                for tr in self.test_scenario.test_runs
            ],
        )
        return output.ExperimentOutput(experiment, output_path)

    def get_run_output(self, job: BaseJob, tr: TestRun, result: JobStatusResult | None = None) -> output_models.Run:
        standalone_job = cast(StandaloneJob, job)
        status: output_models.Status = "running"
        metrics: list[output_models.Metric] = []
        if result is not None:
            status = "completed" if result.is_successful else "failed"
            if job.terminated_by_dependency:
                status = "cancelled"
            if result.is_successful:
                try:
                    observations = tr.test.metric_observations(self.system, tr)
                    metrics = [self._metric_output(observation) for observation in observations]
                except Exception as exc:
                    logging.warning("Cannot extract output metrics for standalone job %s: %s", job.id, exc)
        return output_models.Run(
            path=str(tr.output_path.absolute()),
            jobid=str(job.id),
            status=status,
            metrics=metrics,
            start=standalone_job.start,
            finish=standalone_job.finish,
            iteration=tr.current_iteration,
            step=tr.step,
        )

    def get_runner_job_status(self, job: BaseJob) -> JobStatusResult:
        standalone_job = cast(StandaloneJob, job)
        if standalone_job.terminated_by_dependency:
            return JobStatusResult(is_successful=True)
        if standalone_job.process is None:
            return JobStatusResult(is_successful=True)
        return_code = standalone_job.process.poll()
        if return_code == 0:
            return JobStatusResult(is_successful=True)
        return JobStatusResult(is_successful=False, error_message=f"Process exited with status {return_code}")

    def on_job_completion(self, job: BaseJob) -> None:
        standalone_job = cast(StandaloneJob, job)
        standalone_job.finish = datetime.datetime.now(datetime.timezone.utc)
        if standalone_job.process is not None:
            standalone_job.process.communicate()

    @staticmethod
    def _metric_output(observation: cloudai.metrics.MetricObservation) -> output_models.Metric:
        dimensions = [
            output_models.Dimension(
                name=cloudai.metrics.dimension_label(key),
                value=str(value),
            )
            for key, value in sorted(observation.dimensions.items())
        ]
        return output_models.Metric(
            name=observation.metric.display_name,
            value=observation.value,
            unit=observation.metric.unit,
            dimensions=dimensions,
        )

    def _submit_test(self, tr: TestRun) -> StandaloneJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        exec_cmd = self.get_cmd_gen_strategy(self.system, tr).gen_exec_command()
        logging.info(f"Executing command for test {tr.name}: {exec_cmd}")
        job_id = 0
        process = None
        start = None
        if self.mode == "run":
            start = datetime.datetime.now(datetime.timezone.utc)
            process = self.cmd_shell.execute(exec_cmd)
            pid = process.pid
            job_id = pid
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(tr.name), command=exec_cmd, stdout="", stderr="", message="Failed to retrieve job ID."
                )
        return StandaloneJob(tr, id=job_id, process=process, start=start)
