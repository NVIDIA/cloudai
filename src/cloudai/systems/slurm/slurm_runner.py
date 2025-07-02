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
import re
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
        self.system = cast(SlurmSystem, system)
        self.cmd_shell = CommandShell()

    def get_job_id(self, stdout: str, stderr: str) -> int | None:
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            return int(match.group(1))

        match = re.search(r"submitted with Job ID (\d+)", stdout)  # NemoLauncher specific
        if match:
            return int(match.group(1))

        return None

    def _submit_test(self, tr: TestRun) -> SlurmJob:
        logging.info(f"Running test: {tr.name}")
        exec_cmd = self.get_cmd_gen_strategy(self.system, tr).gen_exec_command(tr)
        logging.debug(f"Executing command for test {tr.name}: {exec_cmd}")
        job_id = 0
        if self.mode == "run":
            stdout, stderr = self.cmd_shell.execute(exec_cmd).communicate()
            job_id = self.get_job_id(stdout, stderr)
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(tr.name),
                    command=exec_cmd,
                    stdout=stdout,
                    stderr=stderr,
                    message="Failed to retrieve job ID.",
                )
        logging.info(f"Submitted slurm job: {job_id}")
        return SlurmJob(tr, id=job_id)

    def on_job_submit(self, tr: TestRun) -> None:
        cmd_gen = self.get_cmd_gen_strategy(self.system, tr)
        cmd_gen.store_test_run(tr)

    def on_job_completion(self, job: BaseJob) -> None:
        logging.debug(f"Job completion callback for job {job.id}")
        self.system.complete_job(cast(SlurmJob, job))
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

    def _get_job_metadata(
        self, job: SlurmJob, steps_metadata: list[SlurmStepMetadata]
    ) -> tuple[Path, SlurmJobMetadata]:
        cmd_gen = cast(SlurmCommandGenStrategy, self.get_cmd_gen_strategy(self.system, job.test_run))
        return job.test_run.output_path / "slurm-job.toml", SlurmJobMetadata(
            job_id=int(job.id),
            name=steps_metadata[0].name,
            state=steps_metadata[0].state,
            exit_code=steps_metadata[0].exit_code,
            start_time=steps_metadata[0].start_time,
            end_time=steps_metadata[0].end_time,
            elapsed_time_sec=steps_metadata[0].elapsed_time_sec,
            job_steps=steps_metadata[1:],
            srun_cmd=cmd_gen.gen_srun_command(job.test_run),
            test_cmd=" ".join(cmd_gen.generate_test_command({}, {}, job.test_run)),
            job_root=job.test_run.output_path.absolute(),
        )

    def store_job_metadata(self, job: SlurmJob):
        system = cast(SlurmSystem, self.system)
        steps_metadata = [self._mock_job_metadata()] if self.mode == "dry-run" else system.get_job_status(job)
        slurm_job_file, job_meta = self._get_job_metadata(job, steps_metadata)

        logging.debug(f"Storing job metadata for job {job.id} to {slurm_job_file}")
        with slurm_job_file.open("w") as job_file:
            toml.dump(job_meta.model_dump(), job_file)
