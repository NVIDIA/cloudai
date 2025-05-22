from typing import cast

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai._core.test_template import TestTemplate
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
                    docker_image_url="docker.io/library/ubuntu:22.04", etcd_endpoint="http://127.0.0.1:2379"
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
            {"docker_image_url": "docker.io/library/ubuntu:22.04", "etcd_endpoint": "http://127.0.0.1:2379", **in_args}
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


def test_gen_nixl_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
    cmd = " ".join(strategy.gen_nixl_srun_command(nixl_bench_tr))
    tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, nixl_bench_tr.test.test_definition)
    assert f"--container-image={tdef.docker_image.installed_path}" in cmd
    assert "--overlap" in cmd
    assert "--ntasks-per-node=1" in cmd
    assert f"--ntasks={nixl_bench_tr.num_nodes}" in cmd
    assert f"-N{nixl_bench_tr.num_nodes}" in cmd


def test_gen_srun_command(nixl_bench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLBenchSlurmCommandGenStrategy(slurm_system, {})
    cmd = strategy._gen_srun_command({}, {}, nixl_bench_tr)
    assert "sleep 5" in cmd
