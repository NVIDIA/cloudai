# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from cloudai import Plugin, Test, TestDefinition, TestRun
from cloudai._core.test import CmdArgs
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


@pytest.fixture
def command_gen_strategy(slurm_system: Mock) -> Generator[SlurmCommandGenStrategy, None, None]:
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, cmd_args)

    with patch.object(strategy, "generate_test_command", return_value=["nccl-test-command"]):
        yield strategy


@pytest.fixture
def test() -> Test:
    return Test(
        test_definition=TestDefinition(
            name="t1",
            description="desc1",
            test_template_name="tt",
            cmd_args=CmdArgs(),
        ),
        test_template=Mock(),
    )


@pytest.fixture
def testrun(tmp_path: Path, test: Test) -> TestRun:
    mock_test_definition = Mock()
    mock_test_template = Mock()

    mock_test_definition.name = "test_job"
    mock_test_definition.extra_cmd_args = ""
    mock_test_template.name = "test_template"

    return TestRun(
        name="test_job",
        test=test,
        output_path=tmp_path,
        num_nodes=2,
        nodes=["node1", "node2"],
        prologue=[],
        epilogue=[],
    )


def test_generate_srun_command_no_plugins(command_gen_strategy: SlurmCommandGenStrategy, testrun: TestRun) -> None:
    srun_command = command_gen_strategy.generate_test_command(
        {},
        {"test_arg": "test_value"},
        testrun.test.extra_cmd_args,
    )
    assert srun_command == ["nccl-test-command"]


def test_prologue_generation(command_gen_strategy: SlurmCommandGenStrategy):
    plugins = [Plugin("Task1", "echo 'Task1'"), Plugin("Task2", "echo 'Task2'")]
    prologue_command = command_gen_strategy._generate_prologue(plugins)
    assert prologue_command == "echo 'Task1'\necho 'Task2'"


def test_epilogue_generation(command_gen_strategy: SlurmCommandGenStrategy):
    plugins = [Plugin("TaskA", "echo 'TaskA'")]
    epilogue_command = command_gen_strategy._generate_epilogue(plugins)
    assert epilogue_command == "echo 'TaskA'"


def read_sbatch_script(sbatch_script_path: Path) -> str:
    with open(sbatch_script_path, "r") as file:
        return file.read()


def assert_sbatch_directives(script_content: str, num_nodes: int, partition: str, nodes: str, output_dir: Path):
    assert "#!/bin/bash" in script_content
    assert "#SBATCH --job-name=" in script_content
    assert f"#SBATCH -N {num_nodes}" in script_content
    assert f"#SBATCH --partition={partition}" in script_content
    assert f"#SBATCH --nodelist={nodes}" in script_content
    assert f"#SBATCH --output={output_dir}/stdout.txt" in script_content
    assert f"#SBATCH --error={output_dir}/stderr.txt" in script_content


def assert_srun_command(script_content: str, mpi_type: str, test_command: str):
    assert f"srun \\\n--mpi={mpi_type}" in script_content
    assert test_command in script_content


def assert_prologue_and_epilogue(script_content: str, prologue_cmd: str, epilogue_cmd: str):
    assert prologue_cmd in script_content
    assert epilogue_cmd in script_content


def test_full_exec_command(command_gen_strategy: SlurmCommandGenStrategy, testrun: TestRun, tmp_path: Path):
    prologue_plugins = [Plugin("PrologueTask1", "echo 'Prologue Task 1'")]
    epilogue_plugins = [Plugin("EpilogueTask1", "echo 'Epilogue Task 1'")]

    testrun.prologue = prologue_plugins
    testrun.epilogue = epilogue_plugins

    sbatch_command = command_gen_strategy.gen_exec_command(testrun)
    assert sbatch_command.startswith("sbatch ")

    sbatch_script_path = tmp_path / "cloudai_sbatch_script.sh"
    assert sbatch_script_path.exists()

    script_content = read_sbatch_script(sbatch_script_path)

    assert_sbatch_directives(
        script_content=script_content, num_nodes=2, partition="main", nodes="node1,node2", output_dir=tmp_path
    )

    assert_srun_command(script_content=script_content, mpi_type="fake-mpi", test_command="nccl-test-command")

    assert_prologue_and_epilogue(
        script_content=script_content, prologue_cmd="echo 'Prologue Task 1'", epilogue_cmd="echo 'Epilogue Task 1'"
    )
