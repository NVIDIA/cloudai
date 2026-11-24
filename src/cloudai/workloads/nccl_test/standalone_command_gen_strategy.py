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

import re
import shlex
from pathlib import Path
from typing import cast

import toml

from cloudai.core import CommandGenStrategy
from cloudai.models.scenario import TestRunDetails

from .nccl import NCCLTestDefinition


class NcclTestStandaloneMPICommandGenStrategy(CommandGenStrategy):
    """Launch nccl-tests directly on assigned hosts with Open MPI."""

    def _command(self) -> str:
        tr = self.test_run
        test = cast(NCCLTestDefinition, tr.test)
        mpi = test.standalone_mpi
        if mpi is None:
            raise ValueError("Standalone NcclTest requires standalone_mpi instead of a container image.")
        if test.cmd_args.use_deepep_matrix:
            raise ValueError("Standalone NcclTest does not support use_deepep_matrix.")
        if not tr.nodes or len(set(tr.nodes)) != len(tr.nodes) or len(tr.nodes) != tr.nnodes:
            raise ValueError("Specify exactly num_nodes distinct MPI hostnames in nodes.")
        if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host) for host in tr.nodes):
            raise ValueError("Use individual MPI hostnames, not host ranges or slot specifications.")

        hosts = ",".join(f"{host}:{mpi.ranks_per_node}" for host in tr.nodes)
        args = [mpi.launcher, "--host", hosts, "-np", str(tr.nnodes * mpi.ranks_per_node)]
        args.extend(["--map-by", f"ppr:{mpi.ranks_per_node}:node", *mpi.launcher_args])
        env = []
        for key, value in self.final_env_vars.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or isinstance(value, list):
                raise ValueError(f"Invalid or unresolved MPI environment setting: {key}")
            env.append(f"{key}={value}")
            args.extend(["-x", key])
        args.extend(self._benchmark_args(test, Path(mpi.binary_dir)))
        return shlex.join(["env", *env, *args])

    @staticmethod
    def _benchmark_args(test: NCCLTestDefinition, binary_dir: Path) -> list[str]:
        subtest = test.cmd_args.subtest_name
        if not isinstance(subtest, str):
            raise ValueError("Resolve subtest_name before generating the MPI command.")
        args = [str(binary_dir / subtest)]
        metadata = {
            "subtest_name",
            "docker_image_url",
            "min_busbw",
            "use_deepep_matrix",
            "alltoallv_matrix_container_path",
        }
        for key, value in test.cmd_args.model_dump(exclude=metadata, exclude_none=True).items():
            if isinstance(value, list):
                raise ValueError(f"Unresolved NCCL argument: {key}")
            args.extend([f"--{key}", str(value)])
        for key, value in test.extra_cmd_args.items():
            args.append(key)
            if value:
                args.append(value)
        return args

    def gen_exec_command(self) -> str:
        command = self._command()
        output = self.test_run.output_path
        full = f"{command} > {shlex.quote(str(output / 'stdout.txt'))} 2> {shlex.quote(str(output / 'stderr.txt'))}"
        output.mkdir(parents=True, exist_ok=True)
        details = TestRunDetails.from_test_run(self.test_run, test_cmd=command, full_cmd=full)
        with (output / self.TEST_RUN_DUMP_FILE_NAME).open("w", encoding="utf-8") as stream:
            toml.dump(details.model_dump(exclude_none=True), stream)
        return full

    def store_test_run(self) -> None:
        self.gen_exec_command()
