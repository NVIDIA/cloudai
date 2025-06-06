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

from cloudai.core import BaseRunner, JobIdRetrievalError, System, TestRun, TestScenario
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

    def _submit_test(self, tr: TestRun) -> StandaloneJob:
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        exec_cmd = tr.test.test_template.gen_exec_command(tr)
        logging.info(f"Executing command for test {tr.name}: {exec_cmd}")
        job_id = 0
        if self.mode == "run":
            pid = self.cmd_shell.execute(exec_cmd).pid
            job_id = tr.test.test_template.get_job_id(str(pid), "")
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=str(tr.name),
                    command=exec_cmd,
                    stdout="",
                    stderr="",
                    message="Failed to retrieve job ID from command output.",
                )
        return StandaloneJob(tr, id=job_id)
