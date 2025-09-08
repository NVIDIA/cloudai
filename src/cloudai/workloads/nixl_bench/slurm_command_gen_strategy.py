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

from typing import cast

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.common.nixl import NIXLCmdGenBase

from .nixl_bench import NIXLBenchTestDefinition


class NIXLBenchSlurmCommandGenStrategy(NIXLCmdGenBase):
    """Command generation strategy for NIXL Bench tests."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)

        self._current_image_url: str | None = None

    def image_path(self) -> str | None:
        return self._current_image_url

    def _container_mounts(self) -> list[str]:
        return []

    @property
    def tdef(self) -> NIXLBenchTestDefinition:
        return cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)

    def _gen_srun_command(self) -> str:
        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                if key == "SLURM_JOB_MASTER_NODE":  # this is an sbatch-level variable, not needed per-node
                    continue
                f.write(f"export {key}={value}\n")

        self._current_image_url = str(self.tdef.docker_image.installed_path)
        etcd_command: list[str] = self.gen_etcd_srun_command(self.tdef.cmd_args.etcd_path)
        self._current_image_url = None
        nixl_commands = self.gen_nixl_srun_commands()

        commands: list[str] = [
            " ".join(etcd_command),
            "etcd_pid=$!",
            " ".join(self.gen_wait_for_etcd_command()),
            *[" ".join(cmd) + " &\nsleep 15" for cmd in nixl_commands[:-1]],
            " ".join(nixl_commands[-1]),
            "kill -9 $etcd_pid",
        ]
        return "\n".join(commands)

    def gen_nixlbench_command(self) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)
        cmd = [tdef.cmd_args.path_to_benchmark]

        for k, v in tdef.cmd_args_dict.items():
            if k == "etcd_endpoints":
                k = "etcd-endpoints"
            cmd.append(f"--{k} {v}")

        return cmd

    def gen_nixl_srun_commands(self) -> list[list[str]]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)
        self._current_image_url = str(tdef.docker_image.installed_path)
        prefix_part = self.gen_srun_prefix()
        self._current_image_url = None

        bash_part = [
            "bash",
            "-c",
            f'"source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            f'{" ".join(self.gen_nixlbench_command())}"',
        ]
        tpn_part = ["--ntasks-per-node=1", "--ntasks=1", "-N1"]

        cmds = [
            [*prefix_part, "--overlap", "--nodelist=$SLURM_JOB_MASTER_NODE", *tpn_part, *bash_part],
        ]

        backend = str(tdef.cmd_args_dict.get("backend", "unset")).upper()
        if backend == "UCX":
            nnodes, _ = self.get_cached_nodes_spec()
            if nnodes > 1:
                cmds = [
                    [*prefix_part, "--overlap", f"--relative={idx}", *tpn_part, *bash_part] for idx in range(nnodes)
                ]
            else:
                cmds *= max(2, nnodes)

        return cmds
