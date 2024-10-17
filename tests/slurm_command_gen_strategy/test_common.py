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
from cloudai import Test, TestDefinition, TestRun, TestTemplate
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
    srun_command = strategy_fixture.generate_srun_command(slurm_args, env_vars, cmd_args, "")

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
