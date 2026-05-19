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

from .ucc import UCCCmdArgs, UCCTestDefinition

_UCC_GEN_MATRIX_CONTAINER = "/opt/hpcx/ucc/tools/perf/generator/input_matrices.txt"


def _ucc_matrix_path_under_deepep_output(dep_out: Path) -> Path | None:
    """DeepEP writes ucc_matrix.txt under a timestamped benchmark subdir; resolve either layout."""
    direct = dep_out / "ucc_matrix.txt"
    if direct.is_file():
        return direct
    nested = sorted(
        dep_out.glob("**/ucc_matrix.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return nested[0] if nested else None


class UCCTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for UCC tests on Slurm systems."""

    def _deepep_ucc_matrix_host_path(self) -> Path | None:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, self.test_run.test)
        if not tdef.cmd_args.use_deepep_matrix:
            return None
        dep_out = deepep_benchmark_root(self.test_run)
        if dep_out is None:
            return None
        return _ucc_matrix_path_under_deepep_output(dep_out)

    def _container_mounts(self) -> List[str]:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, self.test_run.test)
        if not tdef.cmd_args.use_deepep_matrix:
            return []

        deepep_root = deepep_benchmark_root(self.test_run)
        if deepep_root is None:
            return []

        matrix_host = self._deepep_ucc_matrix_host_path()
        if matrix_host is None:
            return []

        return [
            f"{matrix_host.resolve()}:{_UCC_GEN_MATRIX_CONTAINER}",
            f"{deepep_root.resolve()}:{DEEPEP_PREV_MOUNT}:ro",
        ]

    def image_path(self) -> str | None:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        tdef: UCCTestDefinition = cast(UCCTestDefinition, self.test_run.test)
        tdef_cmd_args: UCCCmdArgs = tdef.cmd_args

        srun_command_parts = ["/opt/hpcx/ucc/bin/ucc_perftest"]
        srun_command_parts.append(f"-c {tdef_cmd_args.collective}")
        srun_command_parts.append(f"-b {tdef_cmd_args.b}")
        srun_command_parts.append(f"-e {tdef_cmd_args.e}")
        if tdef_cmd_args.gen is not None:
            srun_command_parts.append(f"--gen {tdef_cmd_args.gen}")
        elif self._deepep_ucc_matrix_host_path() is not None:
            srun_command_parts.append(f"--gen file:name={_UCC_GEN_MATRIX_CONTAINER}")
        srun_command_parts.append("-m cuda")
        srun_command_parts.append("-F")

        if self.test_run.test.extra_cmd_args:
            srun_command_parts.append(self.test_run.test.extra_args_str)

        return srun_command_parts
