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

from typing import Union, List, Dict, Literal

from pydantic import Field

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


def cli_field(*args, cmdline: str, **kwargs):
    """ Wrapper over pydantic Field to add a cmdline attribute to the field. """
    return Field(*args, json_schema_extra={"cmdline": cmdline})


class OSUBenchCmdArgs(CmdArgs):
    """Command line arguments for a OSU Benchmark test."""

    # https://github.com/forresti/osu-micro-benchmarks/blob/master/mpi/collective/osu_coll.c#L202

    docker_image_url: str
    location: str
    """ Location of the OSU Benchmark binary inside the container. """

    benchmark: str
    """ Name of the benchmark to run. """

    message_len: Union[int, List[int]] = cli_field(default=1024, cmdline="-m")
    """ Message length for the benchmark. """

    iterations: int = cli_field(default=10, cmdline="-n")
    """ Number of iterations for the benchmark. """

    target: Literal["cpu", "gpu", "both"] = cli_field(default="cpu", cmdline="-r")
    """ Target for the benchmark. """

    #cuda: bool = cli_field(default=False, cmdline="-cuda")

    @classmethod
    def get_cmdline(cls, name: str) -> str:
        return cls.model_fields[name].json_schema_extra["cmdline"]

    def get_args(self) -> Dict[str, str]:
        """ Retrieve the command line arguments for the OSU benchmark. """

        general = ("docker_image_url", "location", "benchmark")
        return {OSUBenchCmdArgs.get_cmdline(name): value for name, value in self.model_dump().items() if name not in general}


class OSUBenchTestDefinition(TestDefinition):
    """Test definition for a OSU Benchmark test."""

    cmd_args: OSUBenchCmdArgs
    _osu_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._osu_image:
            self._osu_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._osu_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos]

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(exclude={"docker_image_url", "path_to_benchmark"})

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)
