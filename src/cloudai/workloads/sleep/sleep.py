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


from cloudai.core import Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

EXIT_CODE_FILE_NAME = "exit_code.txt"


class SleepCmdArgs(CmdArgs):
    """Sleep test command arguments."""

    docker_image_url: str = "ubuntu:22.04"
    seconds: int = 5


class SleepTestDefinition(TestDefinition):
    """Test object for Sleep."""

    cmd_args: SleepCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        exit_code_path = tr.output_path / EXIT_CODE_FILE_NAME
        if not exit_code_path.is_file():
            # Not every system's command-gen strategy captures an exit code yet
            # (currently only standalone does); fall back to the previous
            # behavior rather than fail runs that never had a chance to produce
            # this file.
            return JobStatusResult(is_successful=True)

        try:
            exit_code_text = exit_code_path.read_text().strip()
        except (OSError, UnicodeDecodeError) as e:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Could not read exit code file {exit_code_path}: {e}.",
            )

        try:
            exit_code = int(exit_code_text)
        except ValueError:
            return JobStatusResult(
                is_successful=False,
                error_message=f"Could not parse exit code from {exit_code_path}: {exit_code_text!r}.",
            )

        if exit_code != 0:
            stderr_path = tr.output_path / "stderr.txt"
            return JobStatusResult(
                is_successful=False,
                error_message=f"sleep exited with code {exit_code}. Check {stderr_path} for details.",
            )

        return JobStatusResult(is_successful=True)
