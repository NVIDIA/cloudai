# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import contextlib
from typing import cast

import pydantic
import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nixl_bench.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition
from cloudai.workloads.nixl_bench.slurm_command_gen_strategy import NIXLBenchSlurmCommandGenStrategy


@pytest.fixture
def nixl_bench_tr(tmp_path) -> TestRun:
    output_path = tmp_path / "nixl-bench"
    output_path.mkdir(parents=True, exist_ok=True)
    return TestRun(
        name="nixl-bench",
        num_nodes=2,
        nodes=[],
        output_path=output_path,
        test=NIXLBenchTestDefinition(
            cmd_args=NIXLBenchCmdArgs(
                docker_image_url="docker.io/library/ubuntu:22.04", path_to_benchmark="./nixlbench"
            ),
            name="nixl-bench",
            description="NIXL Bench",
            test_template_name="NIXLBench",
        ),
    )


class TestNIXLBenchCommand:
    def test_default(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
        cmd = strategy.gen_nixlbench_command()
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
        assert cmd == ["./nixlbench", "--etcd-endpoints=http://$NIXL_ETCD_ENDPOINTS"]
        assert tdef.uses_etcd
        assert not tdef.uses_asio
        assert "NIXL_ASIO_ADDRESS" not in strategy.final_env_vars

    def test_can_set_any_cmd_arg(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        in_args = {"backend": "MPI", "dashed-opt": "DRAM", "under_score_opt": "VRAM"}
        cmd_args = NIXLBenchCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/library/ubuntu:22.04",
                "path_to_benchmark": "/p",
                **in_args,
            }
        )
        nixl_bench_tr.test.cmd_args = cmd_args
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

        cmd = " ".join(strategy.gen_nixlbench_command())

        for k, v in in_args.items():
            assert f"--{k}={v}" in cmd

    def test_asio_runtime_args(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
        tdef.cmd_args.runtime_type = "ASIO"
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

        assert strategy.gen_nixlbench_command() == [
            "./nixlbench",
            "--runtime_type=ASIO",
            "--asio_address=$NIXL_ASIO_ADDRESS",
            "--asio_port=12345",
        ]
        assert not tdef.uses_etcd
        assert tdef.uses_asio
        assert "NIXL_ASIO_ADDRESS" in strategy.final_env_vars

    def test_container_mounts(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        nixl_bench_tr.test.cmd_args = NIXLBenchCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/library/ubuntu:22.04",
                "path_to_benchmark": "/nixlbench",
                "backend": "GUSLI",
                "device_list": "11:K:/dev/nvme0n1,12:F:/p1/store0.bin,13:F:/p2/store0.bin",
                "total_buffer_size": "1kb",
                "filepath": "data",  # also tests this path normalization
            }
        )
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
        assert strategy.gen_nixlbench_command() == [
            "/nixlbench",
            "--filepath=/data",
            "--total_buffer_size=1024",
            "--device_list=11:K:/dev/nvme0n1,12:F:/p1/store0.bin,13:F:/p2/store0.bin",
            "--etcd-endpoints=http://$NIXL_ETCD_ENDPOINTS",
            "--backend=GUSLI",
        ]

        assert strategy.container_mounts() == [
            f"{nixl_bench_tr.output_path}:/cloudai_run_results",
            f"{nixl_bench_tr.output_path.parent}/install:/cloudai_install",
            f"{nixl_bench_tr.output_path}",
            f"{nixl_bench_tr.output_path}/filepath_mount/data:/data",
            f"{nixl_bench_tr.output_path}/device_list_mounts/store0.bin:/p1/store0.bin",
            f"{nixl_bench_tr.output_path}/device_list_mounts/store0_1.bin:/p2/store0.bin",
        ]

        assert (nixl_bench_tr.output_path / "filepath_mount" / "data").is_dir()

        for local_device_filename in ("store0.bin", "store0_1.bin"):
            assert (nixl_bench_tr.output_path / "device_list_mounts" / local_device_filename).is_file()
            assert (nixl_bench_tr.output_path / "device_list_mounts" / local_device_filename).stat().st_size == 1024

    def test_cleanup_job_artifacts(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        nixl_bench_tr.test.cmd_args = NIXLBenchCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/library/ubuntu:22.04",
                "path_to_benchmark": "/nixlbench",
                "backend": "GUSLI",
                "device_list": "11:K:/dev/nvme0n1,12:F:/p1/store0.bin,13:F:/p2/store0.bin",
                "filepath": "/data",
            }
        )
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
        filepath_dir = nixl_bench_tr.output_path / "filepath_mount"
        device_list_dir = nixl_bench_tr.output_path / "device_list_mounts"
        other_file = nixl_bench_tr.output_path / "keep.txt"
        filepath_dir.mkdir(parents=True, exist_ok=True)
        device_list_dir.mkdir(parents=True, exist_ok=True)
        (filepath_dir / "a.txt").write_text("x")
        (device_list_dir / "b.txt").write_text("x")
        other_file.write_text("keep")

        strategy.cleanup_job_artifacts()

        assert not filepath_dir.exists()
        assert not device_list_dir.exists()
        assert other_file.exists()

    @pytest.mark.parametrize(
        ("override", "expected_error_match", "expected_total_buffer_size"),
        (
            ({}, None, None),
            ({"device_list": "11:F:/store-_0.bin", "total_buffer_size": "8gb"}, None, str(8 * 2**30)),
            ({"device_list": "11:F:/store0.bin", "total_buffer_size": "8ggb"}, "total_buffer_size", None),
            ({"device_list": "11:F:/store0.bin", "total_buffer_size": "1024"}, None, "1024"),
            ({"device_list": "11:F:/store0.bin", "total_buffer_size": 1024}, None, "1024"),
            ({"device_list": "11:FF:/store0.bin"}, "Invalid device spec", None),
            ({"device_list": "11:K:/store0.bin,12:K:/store0.bin"}, None, None),
            (
                {"device_list": ["11:K:/store0.bin", "11:F:/store0.bin"], "total_buffer_size": "8gb"},
                None,
                str(8 * 2**30),
            ),
            (
                {
                    "device_list": ["11:K:/store0.bin", "11:F:/store0.bin"],
                    "total_buffer_size": ["8gb", 8000000, 1],
                },
                None,
                [str(8 * 2**30), "8000000", "1"],
            ),
        ),
    )
    def test_device_list_validation(
        self,
        override: dict,
        expected_error_match: str | None,
        expected_total_buffer_size: str | list[str] | None,
    ):
        if expected_error_match is None:
            context = contextlib.nullcontext()
        else:
            context = pytest.raises(pydantic.ValidationError, match=expected_error_match)

        with context:
            cmd_args = NIXLBenchCmdArgs.model_validate(
                {
                    "docker_image_url": "docker.io/library/ubuntu:22.04",
                    "path_to_benchmark": "/p",
                    "backend": "GUSLI",
                }
                | override
            )
            assert cmd_args.total_buffer_size == expected_total_buffer_size


def test_get_etcd_srun_command_with_etcd_image(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    tdef.cmd_args.etcd_image_url = "docker.io/library/etcd:latest"

    cmd = " ".join(strategy.gen_etcd_srun_command(tdef.cmd_args.etcd_path))
    assert tdef.etcd_image is not None
    assert f"--container-image={tdef.etcd_image.installed_path}" in cmd


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
    nixl_bench_tr.test.cmd_args.backend = backend
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    cmds = strategy.gen_nixlbench_srun_commands(strategy.gen_nixlbench_command(), backend)
    assert len(cmds) == exp_ntasks

    for idx, cmd in enumerate(cmds):
        assert "--overlap" in cmd
        assert "--ntasks-per-node=1" in cmd
        assert "--ntasks=1" in cmd
        assert "-N1" in cmd
        if backend == "UCX":
            if nnodes > 1:
                assert f"--nodelist=$(scontrol show hostname $SLURM_JOB_NODELIST | sed -n '{idx + 1}p')" in cmd
                assert "--relative" not in cmd
            else:
                assert "--relative" not in cmd
                assert "--nodelist=$SLURM_JOB_MASTER_NODE" in cmd


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_asio_srun_lifecycle(nixl_bench_tr: TestRun, slurm_system: SlurmSystem, num_nodes: int) -> None:
    nixl_bench_tr.num_nodes = num_nodes
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    tdef.cmd_args.runtime_type = "ASIO"
    nixl_bench_tr.test.cmd_args.backend = "UCX"
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    command = strategy.gen_srun_command()

    assert "NIXL_ASIO_ADDRESS" in strategy.final_env_vars
    assert command.count("nixlbench --runtime_type=ASIO") == 2
    assert "--asio_address=$NIXL_ASIO_ADDRESS" in command
    assert "etcd_pid" not in command
    assert "until curl" not in command
    assert "sleep 4" in command
    assert "sleep 15" not in command
    if num_nodes == 1:
        assert command.count("--nodelist=$SLURM_JOB_MASTER_NODE") == 2
    else:
        assert "sed -n '1p'" in command
        assert "sed -n '2p'" in command


def test_storage_backend_without_runtime(nixl_bench_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    nixl_bench_tr.test.cmd_args.backend = "POSIX"
    tdef.cmd_args.etcd_endpoints = ""
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    command = strategy.gen_srun_command()

    assert command.count("nixlbench") == 1
    assert "--etcd-endpoints" not in command
    assert "etcd_pid" not in command
    assert "until curl" not in command


@pytest.mark.parametrize(
    ("system_args", "test_args", "nodes"),
    [
        (None, "--ntasks=4 --ntasks-per-node=1", []),
        ("-n 4 --ntasks-per-node 1", None, ["node-[033-036]"]),
        ("--ntasks=8", "-n4 --ntasks-per-node=1", []),
        (None, "--ntasks-per-node=1", ["node-[033-036]"]),
    ],
)
def test_independent_storage_placement(
    nixl_bench_tr: TestRun,
    slurm_system: SlurmSystem,
    system_args: str | None,
    test_args: str | None,
    nodes: list[str],
) -> None:
    nixl_bench_tr.num_nodes = 4
    nixl_bench_tr.nodes = nodes
    nixl_bench_tr.extra_srun_args = test_args
    slurm_system.extra_srun_args = system_args
    slurm_system.ntasks_per_node = 8
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    nixl_bench_tr.test.cmd_args.backend = "POSIX"
    tdef.cmd_args.etcd_endpoints = ""

    command = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr).gen_srun_command()

    assert command.startswith("srun -N4 --ntasks-per-node=1 ")
    assert command.count("srun ") == 1
    assert "--nodelist=$SLURM_JOB_MASTER_NODE" not in command
    assert "--ntasks=1" not in command
    assert "-N1" not in command
    assert "sleep " not in command
    assert "etcd" not in command
    for args in (system_args, test_args):
        if args:
            assert args in command
    if system_args and test_args:
        assert command.index(system_args) < command.index(test_args)
    assert (nixl_bench_tr.output_path / "nixlbench.sh").is_file()


@pytest.mark.parametrize("runtime", ["null", "managed", "external", "ASIO"])
def test_existing_placement_is_preserved(nixl_bench_tr: TestRun, slurm_system: SlurmSystem, runtime: str) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    nixl_bench_tr.test.cmd_args.backend = "UCX" if runtime == "ASIO" else "POSIX"
    nixl_bench_tr.num_nodes = 2 if runtime == "ASIO" else 4
    nixl_bench_tr.extra_srun_args = "--ntasks=4 --ntasks-per-node=1"
    if runtime == "null":
        tdef.cmd_args.etcd_endpoints = ""
        nixl_bench_tr.extra_srun_args = "--cpu-bind=none"
    elif runtime == "external":
        tdef.cmd_args.etcd_endpoints = "http://etcd.example:2379"
    elif runtime == "ASIO":
        tdef.cmd_args.runtime_type = "ASIO"

    command = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr).gen_srun_command()

    assert "--ntasks=1 -N1" in command
    assert not (nixl_bench_tr.output_path / "nixlbench.sh").exists()
    if runtime == "managed":
        assert "--ntasks=1 --nodelist=$SLURM_JOB_MASTER_NODE -N1 etcd" in command


def test_managed_etcd_lifecycle(nixl_bench_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    nixl_bench_tr.test.cmd_args.backend = "UCX"
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    command = strategy.gen_srun_command()

    assert tdef.uses_etcd
    assert "--etcd-endpoints=http://$NIXL_ETCD_ENDPOINTS" in command
    assert "etcd_pid=$!" in command
    assert "until curl" in command
    assert "kill -TERM $etcd_pid" in command


def test_external_etcd_is_passed_through(nixl_bench_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    nixl_bench_tr.test.cmd_args.backend = "UCX"
    tdef.cmd_args.etcd_endpoints = "http://etcd.example:2379"
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    command = strategy.gen_srun_command()

    assert not tdef.uses_etcd
    assert "--etcd-endpoints=http://etcd.example:2379" in command
    assert "etcd_pid" not in command
    assert "until curl" not in command


def test_asio_does_not_install_custom_etcd_image(nixl_bench_tr: TestRun) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    tdef.cmd_args.runtime_type = "ASIO"
    tdef.cmd_args.etcd_image_url = "docker.io/library/etcd:latest"

    assert tdef.etcd_image not in tdef.installables


def test_asio_rejects_non_pairwise_process_shape(nixl_bench_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tdef = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    tdef.cmd_args.runtime_type = "ASIO"
    nixl_bench_tr.test.cmd_args.backend = "POSIX"
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)

    with pytest.raises(ValueError, match="ASIO runtime requires exactly two NIXLBench processes"):
        strategy.gen_srun_command()
