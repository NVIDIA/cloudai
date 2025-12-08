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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


def cli_field(*args, cmdline: str, **kwargs):
    """Wrap a pydantic Field to add a cmdline attribute."""
    return Field(*args, json_schema_extra={"cmdline": cmdline}, **kwargs)


class OSUBenchCmdArgs(CmdArgs):
    """Command line arguments for a OSU Benchmark test."""

    docker_image_url: str
    """URL of the Docker image to use for the test."""

    location: str
    """Location of the OSU Benchmark binary inside the container. """

    benchmark: Union[str, List[str]]
    """Name of the benchmark to run. """

    message_size: Optional[Union[str, List[str]]] = cli_field(default=None, cmdline="-m")
    """Message size for the benchmark.

    Examples:
    -m 128    // min = default, max = 128
    -m 2:128  // min = 2, max = 128
    -m 2:     // min 2, max = default
    """

    iterations: Optional[int] = cli_field(default=None, cmdline="-i")
    """Number of iterations for the benchmark."""

    warmup: Optional[int] = cli_field(default=None, cmdline="-x")
    """Number of warmup iterations to skip before timing."""

    mem_limit: Optional[int] = cli_field(default=None, cmdline="-M")
    """Per-process maximum memory consumption in bytes."""

    @classmethod
    def get_cmdline(cls, field: str) -> str:
        """Retrieve the command line argument for the given field."""
        field_info = cls.model_fields[field]

        if not field_info.json_schema_extra or "cmdline" not in field_info.json_schema_extra:
            raise ValueError(f"Field '{field}' does not have a cmdline attribute")

        return field_info.json_schema_extra["cmdline"]

    def get_args(self) -> Dict[str, Any]:
        """Retrieve the command line arguments for the OSU benchmark."""
        general = ("docker_image_url", "location", "benchmark")
        arguments = {field: value for field, value in self.model_dump().items() if field not in general}
        return {OSUBenchCmdArgs.get_cmdline(field): value for field, value in arguments.items() if value is not None}


class OSUBenchTestDefinition(TestDefinition):
    """Test definition for OSU Benchmark test."""

    cmd_args: OSUBenchCmdArgs
    _osu_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._osu_image:
            self._osu_image = DockerImage(url=self.cmd_args.docker_image_url)

        return self._osu_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(exclude={"docker_image_url", "location"})

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stdout_path = tr.output_path / "stdout.txt"
        stderr_path = tr.output_path / "stderr.txt"

        if not stdout_path.is_file():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"stdout.txt file not found in the specified output directory {tr.output_path}. "
                    "This file is expected to be created as a result of the OSU Benchmark test run."
                )
            )

        with open(stdout_path, "r") as f:
            content = f.read()

        if not content.strip():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"stdout.txt file is empty in the specified output directory {tr.output_path}. "
                    f"Please check for fatal errors in {stderr_path}"
            ))

        if "# Size       Avg Latency(us)   Min Latency(us)   Max Latency(us)  Iterations" not in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Expected OSU benchmark output marker not found in stdout.txt in {tr.output_path}. "
                    f"Check for errors in the execution or for a different output format."
                ),
            )

        return JobStatusResult(is_successful=True)
