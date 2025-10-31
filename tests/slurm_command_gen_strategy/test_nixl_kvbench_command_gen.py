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
        "--etcd_endpoints http://$NIXL_ETCD_ENDPOINTS",
    ]
