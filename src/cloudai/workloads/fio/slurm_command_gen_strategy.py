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

from __future__ import annotations

import shlex
from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .command import build_fio_command_parts
from .fio import FioCmdArgs, FioTestDefinition


class FioSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for fio on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        return []

    def image_path(self) -> str | None:
        tdef = cast(FioTestDefinition, self.test_run.test)
        image = tdef.docker_image
        return str(image.installed_path) if image else None

    def gen_srun_prefix(self, use_pretest_extras: bool = False, with_num_nodes: bool = True) -> list[str]:
        prefix = super().gen_srun_prefix(use_pretest_extras, with_num_nodes)
        return [part for part in prefix if not part.startswith("--mpi=")]

    def _fio_task_args(self) -> list[str]:
        args = cast(FioCmdArgs, self.test_run.test.cmd_args)
        task_args: list[str] = []
        num_nodes, _ = self.get_cached_nodes_spec()
        if args.num_tasks_per_node is not None:
            task_args.append(f"--ntasks-per-node={args.num_tasks_per_node}")
            task_args.append(f"--ntasks={num_nodes * args.num_tasks_per_node}")
        return task_args

    def _gen_srun_command(self) -> str:
        srun_command_parts = [*self.gen_srun_prefix(use_pretest_extras=True), *self._fio_task_args()]
        nsys_command_parts = self.gen_nsys_command()
        test_command_parts = self.generate_test_command()

        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                f.write(f'export {key}="{value}"\n')

        full_test_cmd = (
            f'bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            + " ".join(nsys_command_parts + test_command_parts)
            + '"'
        )

        return " ".join(srun_command_parts) + " " + full_test_cmd

    def generate_test_command(self) -> list[str]:
        args = cast(FioCmdArgs, self.test_run.test.cmd_args)
        return build_fio_command_parts(args)

    def gen_srun_success_check(self) -> str:
        stdout = self.test_run.output_path / "stdout.txt"
        return f"grep -Eq 'IOPS=.*BW=|BW=.*IOPS=' {shlex.quote(str(stdout))} && echo 1 || echo 0"
