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
        assert cmd == ["./nixlbench", f"--etcd-endpoints={tdef.cmd_args.etcd_endpoints}"]

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
            f"--etcd-endpoints={nixl_bench_tr.test.cmd_args.etcd_endpoints}",
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

    def test_gen_srun_command_excludes_cleanup(self, nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
        nixl_bench_tr.test.cmd_args = NIXLBenchCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/library/ubuntu:22.04",
                "path_to_benchmark": "/nixlbench",
                "backend": "GUSLI",
                "device_list": "11:F:/store0.bin",
                "filepath": "/data",
            }
        )
        strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
        cmd = strategy._gen_srun_command()

        assert "rm -rf " not in cmd

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


def test_gen_etcd_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    cmd = " ".join(strategy.gen_etcd_srun_command(tdef.cmd_args.etcd_path))
    assert (
        f"{tdef.cmd_args.etcd_path} --listen-client-urls=http://0.0.0.0:2379 --advertise-client-urls=http://$SLURM_JOB_MASTER_NODE:2379"
        " --listen-peer-urls=http://0.0.0.0:2380 --initial-advertise-peer-urls=http://$SLURM_JOB_MASTER_NODE:2380"
        ' --initial-cluster="default=http://$SLURM_JOB_MASTER_NODE:2380"'
    ) in cmd

    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test)
    assert "--overlap" in cmd
    assert "--ntasks-per-node=1" in cmd
    assert "--ntasks=1" in cmd
    assert "--nodelist=$SLURM_JOB_MASTER_NODE" in cmd
    assert "-N1" in cmd
    assert "--container-image=" not in cmd
    assert "--container-mounts" not in cmd

    strategy._current_image_url = str(tdef.docker_image.installed_path)
    cmd = " ".join(strategy.gen_etcd_srun_command(tdef.cmd_args.etcd_path))
    assert f"--container-image={tdef.docker_image.installed_path}" in cmd
    assert "--container-mounts" in cmd


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
                assert f"--relative={idx}" in cmd
            else:
                assert "--relative" not in cmd
                assert "--nodelist=$SLURM_JOB_MASTER_NODE" in cmd


def test_gen_wait_for_etcd_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    cmd = strategy.gen_wait_for_etcd_command()
    assert cmd == [
        "timeout",
        "60",
        "bash",
        "-c",
        '"until curl -s $NIXL_ETCD_ENDPOINTS/health > /dev/null 2>&1; do sleep 1; done" || {\n',
        '  echo "ETCD ($NIXL_ETCD_ENDPOINTS) was unreachable after 60 seconds";\n',
        "  exit 1\n",
        "}",
    ]


def test_gen_kill_and_wait_cmd(nixl_bench_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, nixl_bench_tr)
    cmd = strategy.gen_kill_and_wait_cmd("PID", timeout=120)
    assert cmd == [
        "kill -TERM $PID\n",
        "timeout",
        "120",
        "bash",
        "-c",
        '"while kill -0 $PID 2>/dev/null; do sleep 1; done" || {\n',
        '  echo "Failed to kill ETCD (pid=$PID) within 120 seconds";\n',
        "  exit 1\n",
        "}",
    ]
