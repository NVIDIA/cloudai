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

from cloudai.core import TestRun
from cloudai.workloads.sleep import SleepCmdArgs, SleepTestDefinition
from cloudai.workloads.sleep.sleep import EXIT_CODE_FILE_NAME


def _sleep_test_run(tmp_path: Path) -> TestRun:
    tdef = SleepTestDefinition(
        name="sleep_test",
        description="Simple sleep test",
        test_template_name="Sleep",
        cmd_args=SleepCmdArgs(seconds=1),
    )
    return TestRun(name="sleep-job", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")


def test_was_run_successful_without_exit_code_file_assumes_success(tmp_path: Path) -> None:
    """Systems that don't yet capture an exit code (Slurm, LSF, Kubernetes) keep the old behavior."""
    tr = _sleep_test_run(tmp_path)

    result = tr.test.was_run_successful(tr)

    assert result.is_successful is True


def test_was_run_successful_with_zero_exit_code(tmp_path: Path) -> None:
    tr = _sleep_test_run(tmp_path)
    tr.output_path.mkdir(parents=True)
    (tr.output_path / EXIT_CODE_FILE_NAME).write_text("0\n")

    result = tr.test.was_run_successful(tr)

    assert result.is_successful is True


def test_was_run_successful_with_nonzero_exit_code(tmp_path: Path) -> None:
    tr = _sleep_test_run(tmp_path)
    tr.output_path.mkdir(parents=True)
    (tr.output_path / EXIT_CODE_FILE_NAME).write_text("1\n")

    result = tr.test.was_run_successful(tr)

    assert result.is_successful is False
    assert "exited with code 1" in result.error_message


def test_was_run_successful_with_unparseable_exit_code(tmp_path: Path) -> None:
    tr = _sleep_test_run(tmp_path)
    tr.output_path.mkdir(parents=True)
    (tr.output_path / EXIT_CODE_FILE_NAME).write_text("not-a-number\n")

    result = tr.test.was_run_successful(tr)

    assert result.is_successful is False
    assert "Could not parse exit code" in result.error_message
