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

from pathlib import Path
from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.workloads.deepep.deepep_combined_report import DEEPEP_PREV_MOUNT, deepep_benchmark_root

from .nccl import NCCLCmdArgs, NCCLTestDefinition

_ALLTOALLV_MATRIX_ENV = "ALLTOALLV_MATRIX_FILE"
_NCCL_TESTS_ALLTOALLV_PERF = "/opt/nccl-tests/build/alltoallv_perf"


def _nccl_cmd_scalar(value: object) -> object:
    if isinstance(value, list):
        return value[0] if value else value
    return value


def _nccl_matrix_path_under_deepep_output(dep_out: Path) -> Path | None:
    """DeepEP writes nccl_matrix.txt under the dependency test output or a timestamped benchmark subdir."""
    direct = dep_out / "nccl_matrix.txt"
    if direct.is_file():
        return direct
    nested = sorted(
        dep_out.glob("**/nccl_matrix.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return nested[0] if nested else None


class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NCCL tests on Slurm systems."""

    def _deepep_nccl_matrix_host_path(self) -> Path | None:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        if not tdef.cmd_args.use_deepep_matrix:
            return None
        root = deepep_benchmark_root(self.test_run)
        if root is None:
            return None
        return _nccl_matrix_path_under_deepep_output(root)

    def _container_mounts(self) -> List[str]:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        if not tdef.cmd_args.use_deepep_matrix:
            return []

        matrix_host = self._deepep_nccl_matrix_host_path()
        if matrix_host is None:
            return []

        dest = tdef.cmd_args.alltoallv_matrix_container_path
        mounts: List[str] = [f"{matrix_host.resolve()}:{dest}"]

        dr = deepep_benchmark_root(self.test_run)
        if dr is not None:
            mounts.append(f"{dr.resolve()}:{DEEPEP_PREV_MOUNT}:ro")
        return mounts

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        env_vars = dict(super().final_env_vars)
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        if tdef.cmd_args.use_deepep_matrix and self._deepep_nccl_matrix_host_path() is not None:
            env_vars[_ALLTOALLV_MATRIX_ENV] = tdef.cmd_args.alltoallv_matrix_container_path
        return env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        super().final_env_vars = value

    def image_path(self) -> str | None:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        if tdef.cmd_args.subtest_name == "alltoallv_perf_mpi":
            a = tdef.cmd_args
            parts: List[str] = [
                _NCCL_TESTS_ALLTOALLV_PERF,
                "-b",
                str(_nccl_cmd_scalar(a.minbytes)),
                "-e",
                str(_nccl_cmd_scalar(a.maxbytes)),
                "-g",
                str(_nccl_cmd_scalar(a.ngpus)),
                "-w",
                str(_nccl_cmd_scalar(a.warmup_iters)),
                "-n",
                str(_nccl_cmd_scalar(a.iters)),
            ]
            if self.test_run.test.extra_cmd_args:
                parts.append(self.test_run.test.extra_args_str)
            return parts

        srun_command_parts = [f"{tdef.cmd_args.subtest_name}"]
        skip_cli = {
            "docker_image_url",
            "subtest_name",
            "use_deepep_matrix",
            "alltoallv_matrix_container_path",
        }
        nccl_test_args = tdef.cmd_args.model_dump().keys()
        for arg in nccl_test_args:
            if arg in skip_cli:
                continue

            value = getattr(tdef.cmd_args, arg)
            if value is None:
                continue

            if len(arg) > 1:
                srun_command_parts.append(f"--{arg} {value}")
            else:
                srun_command_parts.append(f"-{arg} {value}")

        if self.test_run.test.extra_cmd_args:
            srun_command_parts.append(self.test_run.test.extra_args_str)

        return srun_command_parts

    def gen_srun_success_check(self) -> str:
        output_file = self.test_run.output_path / "stdout.txt"
        return f'grep -q "Avg bus bandwidth" {output_file} && echo 1 || echo 0'
