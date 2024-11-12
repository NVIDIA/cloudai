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
from unittest.mock import Mock, create_autospec

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
    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, testrun_fixture)
    srun_command = strategy_fixture._gen_srun_command(slurm_args, env_vars, cmd_args, testrun_fixture)

    sbatch_command = strategy_fixture._write_sbatch_script(slurm_args, env_vars, srun_command, testrun_fixture)
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
    tr = Mock(spec=TestRun)
    tr.nodes = ["node1", "node2"]
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == len(tr.nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    tr = create_autospec(TestRun)
    tr.nodes = []
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == tr.num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    # nodes =
    tr = create_autospec(TestRun)
    tr.num_nodes = 0
    tr.nodes = ["node1", "node2"]

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == len(tr.nodes)


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
    "pre_test,post_test,expected_script_lines",
    [
        # No pre_test, no post_test
        (None, None, ["srun"]),
        # One pre_test, no post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            None,
            [
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No pre_test, one post_test
        (
            None,
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "srun",
                "post_test",
            ],
        ),
        # One pre_test, one post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "fi",
            ],
        ),
        # Multiple pre_tests, multiple post_tests
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "    post_test",
                "fi",
            ],
        ),
        # Multiple pre_tests, no post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            None,
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No pre_test, multiple post_tests
        (
            None,
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "srun",
                "post_test",
                "post_test",
            ],
        ),
        # Multiple pre_tests, single post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock()))],
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "fi",
            ],
        ),
    ],
)
def test_pre_test_post_test_combinations(
    strategy_fixture: SlurmCommandGenStrategy,
    testrun_fixture: TestRun,
    pre_test,
    post_test,
    expected_script_lines,
):
    testrun_fixture.pre_test = Mock(spec=TestScenario) if pre_test else None
    testrun_fixture.post_test = Mock(spec=TestScenario) if post_test else None

    if pre_test is not None:
        testrun_fixture.pre_test = Mock(spec=TestScenario)
        testrun_fixture.pre_test.test_runs = pre_test
        for idx, run in enumerate(pre_test):
            run.test.test_template.gen_srun_success_check.return_value = "pre_test"
            run.test.test_template.gen_srun_command.return_value = "srun"
            run.test.name = f"test{idx+1}"

    if post_test is not None:
        testrun_fixture.post_test = Mock(spec=TestScenario)
        testrun_fixture.post_test.test_runs = post_test
        for idx, run in enumerate(post_test):
            run.test.test_template.gen_srun_command.return_value = "post_test"
            run.test.name = f"test{idx+1}"

    sbatch_command = strategy_fixture.gen_exec_command(testrun_fixture)
    script_file_path = sbatch_command.split()[-1]

    with open(script_file_path, "r") as script_file:
        script_content = script_file.read()

    for expected_line in expected_script_lines:
        assert expected_line in script_content, f"Expected '{expected_line}' in generated script but it was missing."
