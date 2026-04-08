# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import toml

from cloudai import TestRun
from cloudai.systems.slurm import SlurmJobMetadata
from cloudai.workloads.megatron_run import MegatronRunCmdArgs, MegatronRunTestDefinition


class TestMegatronRunSuccessCheck:
    def setup_method(self) -> None:
        self.megatron_tdef = MegatronRunTestDefinition(
            name="m",
            description="d",
            test_template_name="MegatronRun",
            cmd_args=MegatronRunCmdArgs(docker_image_url="http://url", run_script=Path(__file__)),
        )

    def _write_slurm_metadata(self, output_path: Path, *, state: str, exit_code: str = "0:0") -> None:
        with (output_path / "slurm-job.toml").open("w", encoding="utf-8") as file:
            toml.dump(
                SlurmJobMetadata(
                    job_id=123,
                    name="megatron",
                    state=state,
                    exit_code=exit_code,
                    start_time="2026-03-22T11:44:22",
                    end_time="2026-03-22T11:54:22",
                    elapsed_time_sec=600,
                    srun_cmd="srun test",
                    test_cmd="python pretrain_gpt.py",
                    job_root=output_path,
                    job_steps=[],
                ).model_dump(),
                file,
            )

    def test_missing_slurm_metadata_fails(self, base_tr: TestRun) -> None:
        result = self.megatron_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "slurm-job.toml file not found" in result.error_message

    def test_failed_slurm_state_fails_even_if_stdout_has_metrics(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        self._write_slurm_metadata(base_tr.output_path, state="FAILED", exit_code="1:0")
        (base_tr.output_path / "stdout.txt").write_text(
            "[2026-01-16 07:32:39] iteration 6/100 | elapsed time per iteration (ms): 15639.0 | "
            "throughput per GPU (TFLOP/s/GPU): 494.6 |\n"
        )

        result = self.megatron_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "state=FAILED" in result.error_message

    def test_completed_slurm_job_with_iteration_metrics_succeeds(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        self._write_slurm_metadata(base_tr.output_path, state="COMPLETED")
        (base_tr.output_path / "stdout.txt").write_text(
            "[2026-01-16 07:32:39] iteration 6/100 | elapsed time per iteration (ms): 15639.0 | "
            "throughput per GPU (TFLOP/s/GPU): 494.6 |\n"
        )

        result = self.megatron_tdef.was_run_successful(base_tr)
        assert result.is_successful

    def test_completed_slurm_job_without_iteration_metrics_fails(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        self._write_slurm_metadata(base_tr.output_path, state="COMPLETED")
        (base_tr.output_path / "stdout.txt").write_text("training started\n")

        result = self.megatron_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "does not contain Megatron iteration metrics" in result.error_message
