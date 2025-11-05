# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal, Optional, Union

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class DDLBCmdArgs(CmdArgs):
    """DDLB test command arguments."""

    docker_image_url: str
    primitive: str
    m: Union[int, list[int]] = 1024
    n: Union[int, list[int]] = 128
    k: Union[int, list[int]] = 1024
    dtype: str
    num_iterations: int = 50
    num_warmups: int = 5
    impl: Union[str, list[str]] = "pytorch;backend=nccl;order=AG_before,AG_after"


class DDLBTestDefinition(TestDefinition):
    """Test object for DDLB."""

    cmd_args: DDLBCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stdout_path = tr.output_path / "stdout.txt"
        if stdout_path.is_file():
            with stdout_path.open("r") as file:
                content = file.read()

                # Check for specific error patterns
                if "Error" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"DDLB test failure detected in {stdout_path}. "
                            "Possible reasons include network errors or remote process exits. "
                            "Please review the DDLB test output and errors in the file first. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )

                # Identify missing success indicators
                if "Benchmark Results" not in content:
                    error_message = (
                        f"Missing success indicators in {stdout_path}: 'Benchmark Results'. "
                        "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
                        "Please review the DDLB test output and errors in the file. "
                        "Ensure the DDLB test ran to completion. You can run the generated sbatch script manually "
                        f"and check if {stdout_path} is created and contains the expected keywords. "
                        "If the issue persists, contact the system administrator."
                    )

                    return JobStatusResult(is_successful=False, error_message=error_message)

                return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stdout.txt file not found in the specified output directory {tr.output_path}. "
                "This file is expected to be created as a result of the DDLB test run. "
                "Please ensure the DDLB test was executed properly and that stdout.txt is generated. "
                f"You can run the generated DDLB test command manually and verify the creation of {stdout_path}. "
                "If the issue persists, contact the system administrator."
            ),
        )
