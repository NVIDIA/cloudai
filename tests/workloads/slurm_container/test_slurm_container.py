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

import pytest
import toml

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmJobMetadata
from cloudai.workloads.slurm_container import SlurmContainerCmdArgs, SlurmContainerTestDefinition


class TestSlurmContainerSuccessCheck:
    def setup_method(self) -> None:
        self.tdef = SlurmContainerTestDefinition(
            name="sc",
            description="desc",
            test_template_name="SlurmContainer",
            cmd_args=SlurmContainerCmdArgs(docker_image_url="docker://url", cmd="bash /scripts/run.sh"),
        )

    def _write_slurm_metadata(self, output_path: Path, *, exit_code: str, state: str = "COMPLETED") -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "slurm-job.toml").open("w", encoding="utf-8") as file:
            toml.dump(
                SlurmJobMetadata(
                    job_id=123,
                    name="sc",
                    state=state,
                    exit_code=exit_code,
                    start_time="2026-07-08T01:36:18",
                    end_time="2026-07-08T01:44:17",
                    elapsed_time_sec=479,
                    srun_cmd="srun test",
                    test_cmd="bash /scripts/run.sh",
                    job_root=output_path,
                    job_steps=[],
                ).model_dump(),
                file,
            )

    def test_missing_slurm_metadata_fails(self, base_tr: TestRun) -> None:
        result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert "slurm-job.toml file not found" in result.error_message

    @pytest.mark.parametrize(
        ("exit_code", "is_successful"),
        [
            ("0:0", True),
            ("0", True),
            ("0:15", True),
            ("1:0", False),
            ("15:0", False),
            ("42:0", False),
            ("137:0", False),
        ],
    )
    def test_exit_code_is_honored(self, base_tr: TestRun, exit_code: str, is_successful: bool) -> None:
        self._write_slurm_metadata(base_tr.output_path, exit_code=exit_code)

        result = self.tdef.was_run_successful(base_tr)

        assert result.is_successful is is_successful
        if not is_successful:
            assert exit_code in result.error_message

    def test_malformed_metadata_is_reported(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        (base_tr.output_path / "slurm-job.toml").write_text("not = valid = toml =", encoding="utf-8")

        result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert "slurm-job.toml" in result.error_message
