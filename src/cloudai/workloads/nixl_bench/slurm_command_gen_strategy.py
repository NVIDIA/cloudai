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

from typing import Any, cast

from cloudai import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .nixl_bench import NIXLBenchTestDefinition


class NIXLBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NIXL Bench tests."""

    def __init__(self, system: SlurmSystem, cmd_args: dict[str, Any]) -> None:
        super().__init__(system, cmd_args)

        self._current_image_url: str | None = None

    def image_path(self, tr: TestRun) -> str | None:
        return self._current_image_url

    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def _gen_srun_command(
        self, env_vars: dict[str, str | list[str]], cmd_args: dict[str, str | list[str]], tr: TestRun
    ) -> str:
        etcd_command: list[str] = self.gen_etcd_srun_command(tr)
        nixl_command: list[str] = self.gen_nixl_srun_command(tr)
        return " ".join(etcd_command) + "\n" + " ".join(nixl_command)

    def gen_etcd_srun_command(self, tr: TestRun) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, tr.test.test_definition)
        self._current_image_url = str(tdef.etcd_image.installed_path)
        etcd_cmd = [
            "/usr/local/bin/etcd",
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            "http://$(hostname -I | awk '{print $1}'):2379",
        ]
        cmd = [
            *self.gen_srun_prefix(tr),
            "--exclusive",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "bash",
            "-c",
            f'"{" ".join(etcd_cmd)}" &',
        ]
        self._current_image_url = None
        return cmd

    def gen_nixl_srun_command(self, tr: TestRun) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, tr.test.test_definition)
        self._current_image_url = str(tdef.docker_image.installed_path)
        nnodes, _ = self.get_cached_nodes_spec(tr)
        nixlbench_cmd = [
            "nixlbench",
            "--etcd-endpoints",
            "http://$SLURM_JOB_MASTER_NODE:2379",
            f"--backend {tdef.cmd_args.backend}",
            f"--initiator_seg_type {tdef.cmd_args.initiator_seg_type}",
        ]
        cmd = [
            *self.gen_srun_prefix(tr),
            "--exclusive",
            "--ntasks-per-node=1",
            f"--ntasks={nnodes}",
            f"-N{nnodes}",
            "bash",
            "-c",
            f'"{" ".join(nixlbench_cmd)}"',
        ]
        self._current_image_url = None
        return cmd
