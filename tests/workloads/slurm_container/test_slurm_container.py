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

from unittest.mock import patch

import pytest

from cloudai.core import TestRun
from cloudai.workloads.slurm_container import SlurmContainerCmdArgs, SlurmContainerTestDefinition
from cloudai.workloads.slurm_container.slurm_container import EXIT_CODE_FILE_NAME


class TestSlurmContainerSuccessCheck:
    def setup_method(self) -> None:
        self.tdef = SlurmContainerTestDefinition(
            name="sc",
            description="desc",
            test_template_name="SlurmContainer",
            cmd_args=SlurmContainerCmdArgs(docker_image_url="docker://url", cmd="bash /scripts/run.sh"),
        )

    def _write_exit_code(self, tr: TestRun, exit_code: str) -> None:
        tr.output_path.mkdir(parents=True, exist_ok=True)
        (tr.output_path / EXIT_CODE_FILE_NAME).write_text(exit_code, encoding="utf-8")

    def test_missing_exit_code_fails(self, base_tr: TestRun) -> None:
        result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert EXIT_CODE_FILE_NAME in result.error_message
        assert "not found" in result.error_message

    @pytest.mark.parametrize(
        ("exit_code", "is_successful"),
        [
            ("0", True),
            ("0\n", True),
            ("1", False),
            ("42", False),
            ("137", False),
        ],
    )
    def test_exit_code_is_honored(self, base_tr: TestRun, exit_code: str, is_successful: bool) -> None:
        self._write_exit_code(base_tr, exit_code)

        result = self.tdef.was_run_successful(base_tr)

        assert result.is_successful is is_successful
        if not is_successful:
            assert exit_code.strip() in result.error_message

    def test_malformed_exit_code_is_reported(self, base_tr: TestRun) -> None:
        self._write_exit_code(base_tr, "not-a-number")

        result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert "Could not parse exit code" in result.error_message

    def test_unreadable_exit_code_is_reported(self, base_tr: TestRun) -> None:
        self._write_exit_code(base_tr, "0")

        with patch("pathlib.Path.read_text", side_effect=PermissionError("permission denied")):
            result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert "Could not read exit code file" in result.error_message

    def test_undecodable_exit_code_is_reported(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        (base_tr.output_path / EXIT_CODE_FILE_NAME).write_bytes(b"\xff\xfe\x00")

        result = self.tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert "Could not read exit code file" in result.error_message
