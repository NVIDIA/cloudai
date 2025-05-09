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

import tarfile
from pathlib import Path

import pytest

from cloudai import Test, TestRun, TestScenario, TestTemplate
from cloudai.reporters.case_reporters import PerTestReporter
from cloudai.reporters.scenario_reporters import TarballReporter
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
def benchmark_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl", description="NCCL test", test_template_name="nccl", cmd_args=NCCLCmdArgs()
    )
    test_template = TestTemplate(system=slurm_system)
    test = Test(test_definition=test_definition, test_template=test_template)
    tr = TestRun(name="benchmark", test=test, num_nodes=1, nodes=["node1"], iterations=3)
    create_test_directories(slurm_system, tr)
    return tr


@pytest.fixture
def dse_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="nccl",
        cmd_args=NCCLCmdArgs(),
        extra_env_vars={"VAR1": ["value1", "value2"]},
        agent_steps=12,
    )
    test_template = TestTemplate(system=slurm_system)
    test = Test(test_definition=test_definition, test_template=test_template)

    tr = TestRun(name="dse", test=test, num_nodes=1, nodes=["node1"], iterations=12)
    create_test_directories(slurm_system, tr)
    return tr


class TestLoadTestTuns:
    def test_load_test_runs_behcnmark_sorted(self, slurm_system: SlurmSystem, benchmark_tr: TestRun) -> None:
        reporter = PerTestReporter(
            slurm_system, TestScenario(name="test_scenario", test_runs=[benchmark_tr]), slurm_system.output_path
        )
        reporter.load_test_runs()

        assert len(reporter.trs) == benchmark_tr.iterations
        for i, tr in enumerate(reporter.trs):
            assert tr.name == benchmark_tr.name
            assert tr.current_iteration == i
            assert tr.step == 0
            assert tr.output_path == slurm_system.output_path / benchmark_tr.name / str(i)

    def test_load_test_runs_dse_sorted(self, slurm_system: SlurmSystem, dse_tr: TestRun) -> None:
        reporter = PerTestReporter(
            slurm_system, TestScenario(name="test_scenario", test_runs=[dse_tr]), slurm_system.output_path
        )
        reporter.load_test_runs()

        assert len(reporter.trs) == dse_tr.test.test_definition.agent_steps * dse_tr.iterations
        for i, tr in enumerate(reporter.trs):
            exp_iter = i // dse_tr.test.test_definition.agent_steps
            exp_step = i % dse_tr.test.test_definition.agent_steps
            assert tr.name == dse_tr.name
            assert tr.current_iteration == exp_iter
            assert tr.step == exp_step
            assert tr.output_path == slurm_system.output_path / dse_tr.name / str(exp_iter) / str(exp_step)


def test_create_tarball_preserves_full_name(tmp_path: Path, slurm_system: SlurmSystem) -> None:
    results_dir = tmp_path / "nemo2.0_llama3_70b_fp8_2025-04-16_14-27-45"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dummy.txt").write_text("test content")

    reporter = TarballReporter(slurm_system, TestScenario(name="dummy", test_runs=[]), results_dir)
    reporter.create_tarball(results_dir)

    tarball_path = tmp_path / "nemo2.0_llama3_70b_fp8_2025-04-16_14-27-45.tgz"
    assert tarball_path.exists()
    assert tarfile.is_tarfile(tarball_path)

    with tarfile.open(tarball_path, "r:gz") as tar:
        assert f"{results_dir.name}/dummy.txt" in tar.getnames()
