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
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem

from .nixl_bench import NIXLBenchTestDefinition


class NIXLBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NIXL Bench tests."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)

        self._current_image_url: str | None = None

    def image_path(self) -> str | None:
        return self._current_image_url

    def _container_mounts(self) -> list[str]:
        return []

    def _gen_srun_command(self) -> str:
        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                if key == "SLURM_JOB_MASTER_NODE":  # this is an sbatch-level variable, not needed per-node
                    continue
                f.write(f"export {key}={value}\n")

        etcd_command: list[str] = self.gen_etcd_srun_command()
        nixl_command: list[str] = self.gen_nixl_srun_command()
        return " ".join(etcd_command) + "\netcd_pid=$!\nsleep 5\n" + " ".join(nixl_command) + "\nkill -9 $etcd_pid\n"

    def gen_etcd_srun_command(self) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)
        self._current_image_url = str(tdef.etcd_image.installed_path)
        etcd_cmd = [
            "/usr/local/bin/etcd",
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            "http://$(hostname -I | awk '{print $1}'):2379",
        ]
        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            "bash",
            "-c",
            f'"{" ".join(etcd_cmd)}" &',
        ]
        self._current_image_url = None
        return cmd

    def gen_nixlbench_command(self) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)
        cmd = [tdef.cmd_args.path_to_benchmark, f"--etcd-endpoints {tdef.cmd_args.etcd_endpoint}"]

        other_args = tdef.cmd_args.model_dump(
            exclude={"docker_image_url", "etcd_endpoint", "path_to_benchmark", "cmd_args"}
        )
        for k, v in other_args.items():
            cmd.append(f"--{k} {v}")

        return cmd

    def gen_nixl_srun_command(self) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test.test_definition)
        self._current_image_url = str(tdef.docker_image.installed_path)
        nnodes, _ = self.get_cached_nodes_spec()
        tpn, ntasks = 1, nnodes
        if nnodes == 1:
            tpn, ntasks = 2, 2
        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            f"--ntasks-per-node={tpn}",
            f"--ntasks={ntasks}",
            f"-N{nnodes}",
            "bash",
            "-c",
            f'"source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            f'{" ".join(self.gen_nixlbench_command())}"',
        ]
        self._current_image_url = None
        return cmd
