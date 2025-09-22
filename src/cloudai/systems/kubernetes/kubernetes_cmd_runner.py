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

from cloudai.core import (
    BaseRunner,
    System,
    TestRun,
    TestScenario,
)
from cloudai.util import CommandShell

from .kubernetes_cmd_job import KubernetesCMDJob
from .kubernetes_cmd_system import KubernetesCMDSystem


class KubernetesCMDRunner(BaseRunner):
    """Implementation of the Runner for Kubernetes CMD-based deployments."""

    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path) -> None:
        super().__init__(mode, system, test_scenario, output_path)
        self.system = cast(KubernetesCMDSystem, system)
        self.cmd_shell = CommandShell()

    def _submit_test(self, tr: TestRun) -> KubernetesCMDJob:
        """
        Execute a given test and returns a job if successful.

        Args:
            tr (TestRun): The test to be executed.

        Returns:
            KubernetesCMDJob: A KubernetesCMDJob object representing the submitted job.
        """
        logging.info(f"Running test: {tr.name}")
        tr.output_path = self.get_job_output_path(tr)
        exec_cmd = self.get_cmd_gen_strategy(self.system, tr).gen_exec_command()
        logging.debug(f"Executing command for test {tr.name}: {exec_cmd}")

        job = KubernetesCMDJob(tr, id=tr.name)

        if self.mode == "run":
            process = self.cmd_shell.execute(exec_cmd)
            job.id = str(process.pid)
            logging.info(f"Started process with PID: {job.id}")

        logging.info(f"Submitted kubernetes job: {job.id}")
        return job
