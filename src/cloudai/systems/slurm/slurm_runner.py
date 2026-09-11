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
import re
from pathlib import Path
from typing import cast

import toml

from cloudai.core import BaseJob, BaseRunner, JobIdRetrievalError, JobStatusResult, System, TestRun, TestScenario
from cloudai.unified_output import ExperimentOutput
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
        self.pinned_nodes: dict[str, list[str]] = {}
        self._experiment: ExperimentOutput | None = None

    def run(self) -> None:
        if self.mode != "run" or any(tr.is_dse_job or tr.step > 0 for tr in self.test_scenario.test_runs):
            super().run()
            return

        try:
            self._experiment = ExperimentOutput(self.test_scenario, self.scenario_root)
        except Exception as exc:
            logging.warning("Cannot initialize unified experiment output: %s", exc)
        completed = False
        try:
            super().run()
            completed = True
        finally:
            if self._experiment is not None:
                self._experiment.finish(self.system, self.jobs, completed)
                self._experiment = None

    def get_job_status(self, job: BaseJob) -> JobStatusResult:
        result = super().get_job_status(job)
        if self._experiment is not None:
            try:
                self._experiment.capture(self.system, job, result)
            except Exception as exc:
                logging.warning("Cannot capture unified output for job %s: %s", job.id, exc)
        return result

    def submit_test(self, tr: TestRun) -> None:
        if tr.pin_nodes and tr.name in self.pinned_nodes:
            tr.nodes = self.pinned_nodes[tr.name].copy()
            logging.info("Forcing test case '%s' to use pinned nodes: %s", tr.name, ",".join(tr.nodes))
        super().submit_test(tr)

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
        cmd_gen = self.get_cmd_gen_strategy(self.system, tr)
        try:
            exec_cmd = cmd_gen.gen_exec_command()
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
        except Exception:
            try:
                cmd_gen.cleanup_job_artifacts()
            except Exception:
                logging.warning(f"Cleanup failed for test run at {tr.output_path}", exc_info=True)
            raise

    def on_job_submit(self, tr: TestRun) -> None:
        cmd_gen = self.get_cmd_gen_strategy(self.system, tr)
        cmd_gen.store_test_run()

    def completed_test_runs(self, job: BaseJob) -> list[TestRun]:
        return [cast(SlurmJob, job).test_run]

    def on_job_completion(self, job: BaseJob) -> None:
        logging.debug(f"Job completion callback for job {job.id}")
        slurm_job = cast(SlurmJob, job)
        slurm_job.nodes = self.system.complete_job(slurm_job)
        self.store_job_metadata(slurm_job)

        tr = slurm_job.test_run
        if self.mode == "run" and tr.pin_nodes and tr.name not in self.pinned_nodes:
            if not slurm_job.nodes:
                logging.error("Cannot pin test case '%s': the job has no recorded node allocation", tr.name)
            else:
                self.pinned_nodes[tr.name] = slurm_job.nodes.copy()
                logging.info("Pinned test case '%s' to nodes: %s", tr.name, ",".join(slurm_job.nodes))

        for tr in self.completed_test_runs(job):
            try:
                self.get_cmd_gen_strategy(self.system, tr).cleanup_job_artifacts()
            except Exception:
                logging.warning(f"Cleanup failed for test run at {tr.output_path}", exc_info=True)

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
            srun_cmd=cmd_gen.gen_srun_command(),
            test_cmd=" ".join(cmd_gen.generate_test_command()),
            job_root=job.test_run.output_path.absolute(),
            nodes=job.nodes,
        )

    def store_job_metadata(self, job: SlurmJob):
        system = cast(SlurmSystem, self.system)
        steps_metadata = [self._mock_job_metadata()] if self.mode == "dry-run" else system.get_job_status(job)
        slurm_job_file, job_meta = self._get_job_metadata(job, steps_metadata)

        logging.debug(f"Storing job metadata for job {job.id} to {slurm_job_file}")
        with slurm_job_file.open("w") as job_file:
            toml.dump(job_meta.model_dump(), job_file)
