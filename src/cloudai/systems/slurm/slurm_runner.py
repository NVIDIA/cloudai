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

import logging
from pathlib import Path
from typing import cast

import toml

from cloudai.core import BaseJob, BaseRunner, JobIdRetrievalError, System, TestRun, TestScenario
from cloudai.util import CommandShell

from .slurm_command_gen_strategy import SlurmCommandGenStrategy
from .slurm_job import SlurmJob
from .slurm_metadata import SlurmJobMetadata, SlurmStepMetadata
from .slurm_system import SlurmSystem


class SlurmRunner(BaseRunner):
    """
    Implementation of the Runner for a system using Slurm.

    Attributes
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path) -> None:
        super().__init__(mode, system, test_scenario, output_path)
        self.cmd_shell = CommandShell()

    def _submit_test(self, tr: TestRun) -> SlurmJob:
        logging.info(f"Running test: {tr.name}")
        exec_cmd = tr.test.test_template.gen_exec_command(tr)
        logging.debug(f"Executing command for test {tr.name}: {exec_cmd}")
        job_id = 0
        if self.mode == "run":
            stdout, stderr = self.cmd_shell.execute(exec_cmd).communicate()
            job_id = tr.test.test_template.get_job_id(stdout, stderr)
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(tr.name),
                    command=exec_cmd,
                    stdout=stdout,
                    stderr=stderr,
                    message="Failed to retrieve job ID from command output.",
                )
        logging.info(f"Submitted slurm job: {job_id}")
        return SlurmJob(tr, id=job_id)

    async def job_completion_callback(self, job: BaseJob) -> None:
        self.store_job_metadata(cast(SlurmJob, job))

    def _mock_job_metadata(self) -> SlurmStepMetadata:
        return SlurmStepMetadata(
            job_id=0,
            step_id="",
            name="unknown",
            state="UNKNOWN",
            exit_code="0",
            start_time="",
            end_time="",
            elapsed_time_sec=0,
            submit_line="dry-run test",
        )

    def store_job_metadata(self, job: SlurmJob):
        system = cast(SlurmSystem, self.system)
        cmd_gen = cast(SlurmCommandGenStrategy, job.test_run.test.test_template.command_gen_strategy)
        res = [self._mock_job_metadata()] if self.mode == "dry-run" else system.get_job_status(job)
        job_meta = SlurmJobMetadata(
            job_id=int(job.id),
            name=res[0].name,
            state=res[0].state,
            exit_code=res[0].exit_code,
            start_time=res[0].start_time,
            end_time=res[0].end_time,
            elapsed_time_sec=res[0].elapsed_time_sec,
            job_steps=res[1:],
            srun_cmd=cmd_gen.gen_srun_command(job.test_run),
            test_cmd=" ".join(cmd_gen.generate_test_command({}, {}, job.test_run)),
        )

        with open(job.test_run.output_path / "slurm-job.toml", "w") as job_file:
            toml.dump(job_meta.model_dump(), job_file)
