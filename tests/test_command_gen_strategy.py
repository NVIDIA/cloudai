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

import pytest

from cloudai.core import CommandGenStrategy, Test, TestRun, TestTemplate
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm.slurm_system import SlurmSystem


class MyCmdGen(CommandGenStrategy):
    def gen_exec_command(self) -> str:
        return "echo 'Hello, World!'"

    def store_test_run(self) -> None:
        pass


@pytest.fixture
def test_run(slurm_system: SlurmSystem) -> TestRun:
    return TestRun(
        name="test_a",
        num_nodes=1,
        nodes=["node1"],
        test=Test(
            test_definition=TestDefinition(name="n", description="d", test_template_name="tt", cmd_args=CmdArgs()),
            test_template=TestTemplate(slurm_system),
        ),
    )


@pytest.mark.parametrize(
    "sys_env,tr_env,expected",
    [
        ({}, {}, {}),
        ({"VAR1": "V1"}, {}, {"VAR1": "V1"}),
        ({"VAR1": "V1"}, {"VAR2": "V2"}, {"VAR1": "V1", "VAR2": "V2"}),
        ({"VAR1": "V1"}, {"VAR1": "V2"}, {"VAR1": "V2"}),
    ],
)
def test_final_env_vars(
    slurm_system: SlurmSystem,
    test_run: TestRun,
    sys_env: dict[str, str],
    tr_env: dict[str, str | list[str]],
    expected: dict[str, str],
):
    slurm_system.global_env_vars = sys_env
    test_run.test.test_definition.extra_env_vars = tr_env
    cmd_gen = MyCmdGen(slurm_system, test_run)

    assert cmd_gen.final_env_vars == expected


def test_final_env_vars_can_override(slurm_system: SlurmSystem, test_run: TestRun):
    slurm_system.global_env_vars = {}
    test_run.test.test_definition.extra_env_vars = {}
    cmd_gen = MyCmdGen(slurm_system, test_run)
    cmd_gen.final_env_vars["VAR1"] = "V1"

    assert cmd_gen.final_env_vars == {"VAR1": "V1"}
