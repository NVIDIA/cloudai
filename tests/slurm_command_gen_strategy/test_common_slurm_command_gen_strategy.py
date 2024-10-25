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
from unittest.mock import Mock

import pytest

from cloudai import Test, TestDefinition, TestRun, TestScenario, TestTemplate
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, cmd_args)
    return strategy


@pytest.fixture
def testrun_fixture(tmp_path: Path) -> TestRun:
    mock_test_definition = Mock(spec=TestDefinition)
    mock_test_template = Mock(spec=TestTemplate)

    mock_test_definition.name = "test_job"
    mock_test_definition.description = "Test description"
    mock_test_definition.cmd_args_dict = {"test_arg": "test_value"}
    mock_test_definition.extra_args_str = "extra_arg"
    mock_test_definition.extra_env_vars = {"TEST_VAR": "VALUE"}

    mock_test_template.name = "test_template"

    test = Test(test_definition=mock_test_definition, test_template=mock_test_template)

    return TestRun(name="test_job", test=test, output_path=tmp_path, num_nodes=2, nodes=["node1", "node2"])


def test_filename_generation(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    slurm_args = strategy_fixture._parse_slurm_args(
        job_name_prefix, env_vars, cmd_args, testrun_fixture.num_nodes, testrun_fixture.nodes
    )
    srun_command = strategy_fixture._gen_srun_command(slurm_args, env_vars, cmd_args, "")

    sbatch_command = strategy_fixture._write_sbatch_script(
        slurm_args, env_vars, srun_command, testrun_fixture.output_path
    )
    filepath_from_command = sbatch_command.split()[-1]

    assert testrun_fixture.output_path.joinpath("cloudai_sbatch_script.sh").exists()

    with open(filepath_from_command, "r") as file:
        file_contents = file.read()

    assert "test_job" in file_contents
    assert "node1,node2" in file_contents
    assert "srun" in file_contents
    assert "--mpi=fake-mpi" in file_contents


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = []
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 0

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_raises_if_no_default_partition(slurm_system: SlurmSystem):
    slurm_system.default_partition = ""
    with pytest.raises(ValueError) as exc_info:
        SlurmCommandGenStrategy(slurm_system, {})
    assert (
        "Default partition not set in the Slurm system object. "
        "The 'default_partition' attribute should be properly defined in the Slurm "
        "system configuration. Please ensure that 'default_partition' is set correctly "
        "in the corresponding system configuration (e.g., system.toml)."
    ) in str(exc_info.value)


@pytest.mark.parametrize(
    "prologue,epilogue,expected_script_lines",
    [
        # No prologue, no epilogue
        (None, None, ["srun"]),
        # One prologue, no epilogue
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            None,
            [
                "SUCCESS_0=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "PROLOGUE_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PROLOGUE_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No prologue, one epilogue
        (
            None,
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "srun",
                "epilogue",
            ],
        ),
        # One prologue, one epilogue
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "SUCCESS_0=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "PROLOGUE_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PROLOGUE_SUCCESS -eq 1 ]; then",
                "    srun",
                "    epilogue",
                "fi",
            ],
        ),
        # Multiple prologues, multiple epilogues
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "SUCCESS_0=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "SUCCESS_1=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "PROLOGUE_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PROLOGUE_SUCCESS -eq 1 ]; then",
                "    srun",
                "    epilogue",
                "    epilogue",
                "fi",
            ],
        ),
        # Multiple prologues, no epilogue
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            None,
            [
                "SUCCESS_0=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "SUCCESS_1=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "PROLOGUE_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PROLOGUE_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No prologue, multiple epilogues
        (
            None,
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "srun",
                "epilogue",
                "epilogue",
            ],
        ),
        # Multiple prologues, single epilogue
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock()))],
            [
                "SUCCESS_0=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "SUCCESS_1=$(grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0)",
                "PROLOGUE_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PROLOGUE_SUCCESS -eq 1 ]; then",
                "    srun",
                "    epilogue",
                "fi",
            ],
        ),
    ],
)
def test_prologue_epilogue_combinations(
    strategy_fixture: SlurmCommandGenStrategy,
    testrun_fixture: TestRun,
    prologue,
    epilogue,
    expected_script_lines,
    tmp_path,
):
    testrun_fixture.prologue = Mock(spec=TestScenario) if prologue else None
    testrun_fixture.epilogue = Mock(spec=TestScenario) if epilogue else None

    if prologue is not None:
        testrun_fixture.prologue = Mock(spec=TestScenario)
        testrun_fixture.prologue.test_runs = prologue
        for idx, run in enumerate(prologue):
            run.test.test_template.gen_srun_success_check.return_value = (
                "grep -q 'Avg bus bandwidth' stdout.txt && echo 1 || echo 0"
            )
            run.test.test_template.gen_srun_command.return_value = "srun"
            run.test.name = f"test{idx+1}"
    else:
        testrun_fixture.prologue = None

    if epilogue is not None:
        testrun_fixture.epilogue = Mock(spec=TestScenario)
        testrun_fixture.epilogue.test_runs = epilogue
        for idx, run in enumerate(epilogue):
            run.test.test_template.gen_srun_command.return_value = "epilogue"
            run.test.name = f"test{idx+1}"
    else:
        testrun_fixture.epilogue = None

    sbatch_command = strategy_fixture.gen_exec_command(testrun_fixture)
    script_file_path = sbatch_command.split()[-1]

    with open(script_file_path, "r") as script_file:
        script_content = script_file.read()

    for expected_line in expected_script_lines:
        assert expected_line in script_content, f"Expected '{expected_line}' in generated script but it was missing."
