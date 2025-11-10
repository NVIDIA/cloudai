# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import cast

from cloudai.core import BaseJob, BaseRunner, TestRun

from .runai_job import RunAIJob
from .runai_system import RunAISystem


class RunAIRunner(BaseRunner):
    """Class to manage and execute workloads using the RunAI platform."""

    def _submit_test(self, tr: TestRun) -> RunAIJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        job_spec = self.get_json_gen_strategy(self.system, tr).gen_json()
        logging.debug(f"Generated JSON for test {tr.name}: {job_spec}")

        if self.mode == "run":
            runai_system = cast(RunAISystem, self.system)
            training = runai_system.create_training(job_spec)
            job = RunAIJob(test_run=tr, id=training.workload_id, status=training.actual_phase)
            logging.info(f"Submitted RunAI job: {job.id}")
            return job
        else:
            raise RuntimeError("Invalid mode for submitting a test.")

    async def job_completion_callback(self, job: BaseJob) -> None:
        runai_system = cast(RunAISystem, self.system)
        job = cast(RunAIJob, job)
        workload_id = str(job.id)
        runai_system.get_workload_events(workload_id, job.test_run.output_path / "events.txt")
        await runai_system.store_logs(workload_id, job.test_run.output_path / "stdout.txt")

    def kill_job(self, job: BaseJob) -> None:
        runai_system = cast(RunAISystem, self.system)
        job = cast(RunAIJob, job)
        runai_system.delete_training(str(job.id))
