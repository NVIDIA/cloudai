from typing import cast

import pytest

from cloudai.core import Test, TestRun, TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nixl_kvbench import (
    NIXLKVBenchCmdArgs,
    NIXLKVBenchSlurmCommandGenStrategy,
    NIXLKVBenchTestDefinition,
)


@pytest.fixture
def kvbench() -> NIXLKVBenchTestDefinition:
    return NIXLKVBenchTestDefinition(
        name="n", description="d", test_template_name="NIXLKVBench", cmd_args=NIXLKVBenchCmdArgs()
    )


@pytest.fixture
def kvbench_tr(slurm_system: SlurmSystem, kvbench: NIXLKVBenchTestDefinition) -> TestRun:
    return TestRun(
        name="nixl-bench",
        num_nodes=2,
        nodes=[],
        test=Test(test_template=TestTemplate(slurm_system), test_definition=kvbench),
    )


@pytest.mark.parametrize(
    "flag,backend,expected",
    [
        (None, "UCX", True),
        (False, "UCX", False),
        (None, "POSIX", None),
        (True, "POSIX", True),
    ],
)
def test_with_etcd(kvbench: NIXLKVBenchTestDefinition, flag: bool | None, backend: str | None, expected: bool | None):
    kvbench.cmd_args = NIXLKVBenchCmdArgs.model_validate({"with_etcd": flag, "backend": backend})
    assert kvbench.cmd_args.with_etcd is expected


def test_gen_kvbench_ucx(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.test_definition.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {"model": "./model.yaml", "model_config": "./cfg.yaml", "backend": "UCX", "source": "src", "op_type": "READ"}
    )
    kvbench = cast(NIXLKVBenchTestDefinition, kvbench_tr.test.test_definition)
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()
    assert cmd == [
        f"{kvbench.cmd_args.python_executable}",
        f"{kvbench.cmd_args.kvbench_script}",
        "--backend UCX",
        "--model ./model.yaml",
        "--model_config ./cfg.yaml",
        "--source src",
        "--op_type READ",
        "--etcd-endpoints http://$SERVER:2379",
    ]


def test_gen_kvbench_posix(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.test_definition.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "model": "./model.yaml",
            "model_config": "./cfg.yaml",
            "backend": "POSIX",
            "num_requests": 1,
            "source": "file",
            "num_iter": 16,
            "page_size": 256,
            "filepath": "/data",
        }
    )
    kvbench = cast(NIXLKVBenchTestDefinition, kvbench_tr.test.test_definition)
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()
    assert cmd == [
        f"{kvbench.cmd_args.python_executable}",
        f"{kvbench.cmd_args.kvbench_script}",
        "--backend POSIX",
        "--model ./model.yaml",
        "--model_config ./cfg.yaml",
        "--num_requests 1",
        "--source file",
        "--num_iter 16",
        "--page_size 256",
        "--filepath /data",
    ]
