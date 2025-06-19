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
def nixl_bench_tr(slurm_system: SlurmSystem):
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
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
        cmd = strategy.gen_nixlbench_command(nixl_bench_tr)
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
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
        nixl_bench_tr.test.test_definition.cmd_args = cmd_args

        cmd = " ".join(strategy.gen_nixlbench_command(nixl_bench_tr))

        for k, v in in_args.items():
            assert f"--{k} {v}" in cmd


def test_gen_etcd_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
    cmd = " ".join(strategy.gen_etcd_srun_command(nixl_bench_tr))
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


@pytest.mark.parametrize("nnodes", (1, 2))
def test_gen_nixl_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem, nnodes: int):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
    nixl_bench_tr.num_nodes = nnodes
    cmd = " ".join(strategy.gen_nixl_srun_command(nixl_bench_tr))

    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test.test_definition)

    exp_tpn, exp_ntasks, exp_nnodes = 1, nnodes, nnodes
    if nnodes == 1:  # at least two processes of nixlbench should run
        exp_tpn, exp_ntasks = 2, 2
    assert f"--container-image={tdef.docker_image.installed_path}" in cmd
    assert "--overlap" in cmd
    assert f"--ntasks-per-node={exp_tpn}" in cmd
    assert f"--ntasks={exp_ntasks}" in cmd
    assert f"-N{exp_nnodes}" in cmd


def test_gen_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
    cmd = strategy._gen_srun_command({}, {}, nixl_bench_tr)
    assert "sleep 5" in cmd
