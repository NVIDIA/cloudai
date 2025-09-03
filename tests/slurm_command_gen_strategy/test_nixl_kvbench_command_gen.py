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
        name="n",
        description="d",
        test_template_name="NIXLKVBench",
        cmd_args=NIXLKVBenchCmdArgs(docker_image_url="docker://image/url"),
    )


@pytest.fixture
def kvbench_tr(slurm_system: SlurmSystem, kvbench: NIXLKVBenchTestDefinition) -> TestRun:
    return TestRun(
        name="nixl-bench",
        num_nodes=2,
        nodes=[],
        test=Test(test_template=TestTemplate(slurm_system), test_definition=kvbench),
    )


def test_gen_kvbench_ucx(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.test_definition.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "model": "./model.yaml",
            "model_config": "./cfg.yaml",
            "backend": "UCX",
            "source": "src",
            "op_type": "READ",
        }
    )
    kvbench = cast(NIXLKVBenchTestDefinition, kvbench_tr.test.test_definition)
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()
    assert cmd == [
        f"{kvbench.cmd_args.python_executable}",
        f"{kvbench.cmd_args.kvbench_script}",
        kvbench.cmd_args.command,
        "--backend UCX",
        "--model ./model.yaml",
        "--model_config ./cfg.yaml",
        "--source src",
        "--op_type READ",
        "--etcd-endpoints http://$NIXL_ETCD_ENDPOINTS:2379",
    ]
