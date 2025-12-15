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

from unittest.mock import Mock

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.bash_cmd import BashCmdArgs, BashCmdCommandGenStrategy, BashCmdTestDefinition


@pytest.fixture
def bash_tr(slurm_system: SlurmSystem) -> TestRun:
    tr = TestRun(
        name="bash",
        test=BashCmdTestDefinition(
            name="bash",
            description="desc",
            test_template_name="t",
            cmd_args=BashCmdArgs(cmd="echo 'Hello, world!'"),
        ),
        num_nodes=1,
        nodes=[],
        output_path=slurm_system.output_path,
    )
    return tr


@pytest.fixture
def bash_cmd_gen(slurm_system: SlurmSystem, bash_tr: TestRun) -> BashCmdCommandGenStrategy:
    return BashCmdCommandGenStrategy(slurm_system, bash_tr)


def test_gen_srun_success_check(bash_cmd_gen: BashCmdCommandGenStrategy):
    res = bash_cmd_gen.gen_srun_success_check()
    assert res == "[ $? -eq 0 ] && echo 1 || echo 0"


def test_generate_test_command(bash_cmd_gen: BashCmdCommandGenStrategy):
    res = bash_cmd_gen.generate_test_command()
    assert res == ["echo 'Hello, world!'"]


def test_gen_srun_prefix(bash_cmd_gen: BashCmdCommandGenStrategy):
    res = bash_cmd_gen.gen_srun_prefix()
    assert res == []


def test_gen_nsys_command(bash_cmd_gen: BashCmdCommandGenStrategy):
    res = bash_cmd_gen.gen_nsys_command()
    assert res == []


def test_gen_container_mounts(bash_cmd_gen: BashCmdCommandGenStrategy):
    res = bash_cmd_gen._container_mounts()
    assert res == []


def test_installables(bash_tr: TestRun):
    res = bash_tr.test.installables
    assert res == []

    bash_tr.test.git_repos = [Mock()]
    res = bash_tr.test.installables
    assert len(res) == 1
