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

from cloudai import Test, TestRun, TestScenario
from cloudai._core.reporter import Reporter
from cloudai._core.test_template import TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


def create_test_directories(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    test_dir = slurm_system.output_path / test_run.name

    for iteration in range(test_run.iterations):
        folder = test_dir / str(iteration)
        folder.mkdir(exist_ok=True, parents=True)
        if test_run.test.test_definition.is_dse_job:
            for step in range(test_run.test.test_definition.agent_steps):
                (folder / str(step)).mkdir(exist_ok=True, parents=True)


@pytest.fixture
def bench_tdef() -> NCCLTestDefinition:
    return NCCLTestDefinition(
        name="regular_test",
        description="Regular test description",
        test_template_name="test_template",
        cmd_args=NCCLCmdArgs(),
    )


@pytest.fixture
def dse_tdef() -> NCCLTestDefinition:
    return NCCLTestDefinition(
        name="dse_test",
        description="DSE test description",
        test_template_name="test_template",
        cmd_args=NCCLCmdArgs(),
        extra_env_vars={"VAR1": ["value1", "value2"]},  # Makes it a DSE job
        agent_steps=2,
    )


@pytest.fixture
def benchmark_tr(slurm_system: SlurmSystem, bench_tdef: NCCLTestDefinition) -> TestRun:
    test_template = TestTemplate(system=slurm_system, name="test_template")
    test_run = TestRun(
        name="regular_test_run",
        test=Test(test_definition=bench_tdef, test_template=test_template),
        num_nodes=1,
        nodes=[],
        iterations=2,
    )
    create_test_directories(slurm_system, test_run)
    return test_run


@pytest.fixture
def dse_tr(slurm_system: SlurmSystem, dse_tdef: NCCLTestDefinition) -> TestRun:
    test_template = TestTemplate(system=slurm_system, name="test_template")
    test_run = TestRun(
        name="dse_test_run",
        test=Test(test_definition=dse_tdef, test_template=test_template),
        num_nodes=1,
        nodes=[],
    )
    create_test_directories(slurm_system, test_run)
    return test_run


def test_load_test_runs_regular(slurm_system: SlurmSystem, benchmark_tr: TestRun) -> None:
    reporter = Reporter(
        slurm_system, TestScenario(name="test_scenario", test_runs=[benchmark_tr]), slurm_system.output_path
    )
    reporter.load_test_runs()

    assert len(reporter.trs) == benchmark_tr.iterations
    for i, tr in enumerate(reporter.trs):  # test_runs should be sorted for iterations and steps
        assert tr.name == benchmark_tr.name
        assert tr.current_iteration == i
        assert tr.step == 0
        assert tr.output_path == slurm_system.output_path / benchmark_tr.name / str(i)


def test_load_test_runs_dse(slurm_system: SlurmSystem, dse_tr: TestRun) -> None:
    reporter = Reporter(slurm_system, TestScenario(name="test_scenario", test_runs=[dse_tr]), slurm_system.output_path)
    reporter.load_test_runs()

    assert len(reporter.trs) == dse_tr.test.test_definition.agent_steps
    for i, tr in enumerate(reporter.trs):  # test_runs should be sorted for iterations and steps
        assert tr.name == dse_tr.name
        assert tr.current_iteration == 0
        assert tr.step == i
        assert tr.output_path == slurm_system.output_path / dse_tr.name / "0" / str(i)
