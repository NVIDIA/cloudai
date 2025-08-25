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

import pytest

from cloudai.core import Test, TestRun, TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nixl_bench.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition
from cloudai.workloads.nixl_bench.slurm_command_gen_strategy import NIXLBenchSlurmCommandGenStrategy


@pytest.fixture
def nixl_bench_tr(slurm_system: SlurmSystem) -> TestRun:
    return TestRun(
        name="nixl-bench",
        num_nodes=2,
        nodes=[],
        test=Test(
            test_template=TestTemplate(slurm_system),
            test_definition=NIXLBenchTestDefinition(
                etcd_image_url="docker.io/library/etcd:3.5.1",
                cmd_args=NIXLBenchCmdArgs(
                    docker_image_url="docker.io/library/ubuntu:22.04",
                    etcd_endpoint="http://127.0.0.1:2379",
                    path_to_benchmark="./nixlbench",
                ),
                name="nixl-bench",
                description="NIXL Bench",
                test_template_name="NIXLBench",
            ),
        ),
    )


class TestNIXLBenchCommand:
    def test_default(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
        cmd = strategy.gen_nixlbench_command()
        assert cmd == ["./nixlbench", "--etcd-endpoints http://127.0.0.1:2379"]

    def test_can_set_any_cmd_arg(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        in_args = {"backend": "MPI", "dashed-opt": "DRAM", "under_score_opt": "VRAM"}
        cmd_args = NIXLBenchCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/library/ubuntu:22.04",
                "etcd_endpoint": "http://127.0.0.1:2379",
                "path_to_benchmark": "/p",
                **in_args,
            }
        )
        nixl_bench_tr.test.test_definition.cmd_args = cmd_args
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

        cmd = " ".join(strategy.gen_nixlbench_command())

        for k, v in in_args.items():
            assert f"--{k} {v}" in cmd


def test_gen_etcd_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    cmd = " ".join(strategy.gen_etcd_srun_command())
    assert (
        "/usr/local/bin/etcd --listen-client-urls http://0.0.0.0:2379 "
        "--advertise-client-urls http://$(hostname -I | awk '{print $1}'):2379"
    ) in cmd

    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test.test_definition)
    assert f"--container-image={tdef.etcd_image.installed_path}" in cmd
    assert "--container-mounts" in cmd
    assert "--overlap" in cmd
    assert "--ntasks-per-node=1" in cmd
    assert "--ntasks=1" in cmd
    assert "--nodelist=$SLURM_JOB_MASTER_NODE" in cmd
    assert "-N1" in cmd


@pytest.mark.parametrize(
    "backend,nnodes,exp_ntasks",
    [
        ("UCX", 1, 2),  # UCX single node requires two processes, both are on the same node
        ("UCX", 2, 2),  # UCX multi node requires two processes, one on each node
        ("OBJ", 1, 1),
        ("GPUNETIO", 1, 1),
        ("GDS", 1, 1),
    ],
)
def test_gen_nixl_srun_command(
    nixl_bench_tr: TestRun, slurm_system: SlurmSystem, backend: str, nnodes: int, exp_ntasks: int
):
    nixl_bench_tr.num_nodes = nnodes
    nixl_bench_tr.test.test_definition.cmd_args.backend = backend
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    cmds = strategy.gen_nixl_srun_commands()
    assert len(cmds) == exp_ntasks

    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test.test_definition)

    for idx, cmd in enumerate(cmds):
        assert f"--container-image={tdef.docker_image.installed_path}" in cmd
        assert "--overlap" in cmd
        assert "--ntasks-per-node=1" in cmd
        assert "--ntasks=1" in cmd
        assert "-N1" in cmd
        if backend == "UCX":
            if nnodes > 1:
                assert f"--relative={idx}" in cmd
            else:
                assert "--relative" not in cmd


def test_gen_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    cmd = strategy._gen_srun_command()
    assert "sleep 5" in cmd
